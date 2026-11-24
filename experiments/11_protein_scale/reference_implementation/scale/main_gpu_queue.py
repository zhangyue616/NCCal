"""Serialized, resumable main GPU dispatch. Scientific outputs stay sealed."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
import sys
sys.dont_write_bytecode = True
import argparse
import ctypes
import gc
import json
import re
import subprocess
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import scale_timing_driver as driver


def utc():
    return datetime.now(timezone.utc).isoformat()


def emit(phase,**values):
    print(json.dumps({"utc":utc(),"phase":phase,**values}),flush=True)


def alive(pid):
    handle = ctypes.windll.kernel32.OpenProcess(0x1000,False,int(pid))
    if handle:
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    return False


def deadline_check(deadline,estimate):
    return datetime.now(timezone.utc)+timedelta(seconds=1.2*float(estimate)) <= deadline


def completed_records():
    records = []
    for index in range(2,200):
        path = HERE/"sealed"/"main_650m"/f"init_{index:03d}"/"COMPLETE.json"
        if path.exists():
            receipt = driver.read(path)
            if receipt["index"] != index or receipt["seed"] != 20261118000+index or receipt["status"] != "COMPLETE":
                raise ValueError("Invalid completed initialization receipt")
            records.append(receipt)
    return records


def progress(status,**more):
    done = completed_records()
    record = {"status":status,"updated_utc":utc(),"PID":os.getpid(),"pilot_reused_once":True,
        "completed_650m_including_pilot":1+len(done),"target_K":199,
        "completed_main_indices":[r["index"] for r in done],"scientific_results_disclosed":False,**more}
    driver.write(HERE/"MAIN_GPU_PROGRESS.json",record)


def run_network(reference,direction,ctx,propensity,index):
    base = HERE/"sealed"/"main_650m"/f"init_{index:03d}"
    base.mkdir(parents=True,exist_ok=True)
    attempts = sorted(p for p in base.glob("attempt_*") if p.is_dir())
    attempt = len(attempts)+1
    if attempt > 3:
        raise ValueError("Same initialization reached the maximum of three attempts")
    folder = base/f"attempt_{attempt:03d}"
    folder.mkdir()
    started = time.perf_counter()
    started_utc = utc()
    driver.write(folder/"STARTED.json",{"index":index,"seed":20261118000+index,
        "attempt":attempt,"started_utc":started_utc,"PID":os.getpid(),
        "runner_sha256":driver.sha(__file__)})
    timings = {}
    torch = reference.torch
    net = None
    emit("NETWORK_START",index=index,attempt=attempt)
    try:
        stage = time.perf_counter()
        # Frozen constructor parameterization only; its code and seed rule are
        # unchanged. No other model is live while these globals are assigned.
        saved_index,saved_seed = driver.INDEX,driver.SEED
        driver.INDEX,driver.SEED = index,20261118000+index
        try:
            net,vocab = driver.make_untrained(reference,folder)
        finally:
            driver.INDEX,driver.SEED = saved_index,saved_seed
        identity = driver.read(folder/"MODEL_IDENTITY.json")
        for previous in attempts:
            prior = previous/"MODEL_IDENTITY.json"
            if prior.exists() and driver.read(prior)["state_sha256"] != identity["state_sha256"]:
                raise ValueError("Same-seed CPU initialization differs from the earlier attempt")
        timings["initialization_hash_and_transfer_seconds"] = time.perf_counter()-stage
        zd,dtiming = driver.extract_blocks(torch,net,vocab,ctx.d,ctx.cohort)
        timings["development_forward"] = dtiming
        emit("D_FORWARD_COMPLETE",index=index,seconds=dtiming["wall_seconds"])
        selections,selected,folders = {},{},{}
        timings["development_selection_seconds_by_block"] = {}
        for block in driver.BLOCKS:
            folders[block] = folder/f"block{block}"
            folders[block].mkdir()
            stage = time.perf_counter()
            selections[block],selected[block] = ctx.select(zd[block],folders[block],f"650m-block{block}")
            timings["development_selection_seconds_by_block"][str(block)] = time.perf_counter()-stage
        lock = {str(folders[b]/"SELECTION_LOCK.json"):driver.sha(folders[b]/"SELECTION_LOCK.json") for b in driver.BLOCKS}
        driver.write(folder/"D_BOTH_BLOCKS_FROZEN.json",{"sealed_utc":utc(),
            "both_locked_before_confirmation":True,"selection_files":lock})
        if not all(driver.sha(Path(p)) == value for p,value in lock.items()):
            raise ValueError("D selection lock changed before confirmation extraction")
        del zd[17]
        axes = {b:int(selections[b]["axis_zero_based"]) for b in driver.BLOCKS}
        zt,ttiming = driver.extract_blocks(torch,net,vocab,ctx.t,ctx.cohort,axes=axes)
        timings["confirmation_forward"] = ttiming
        timings["confirmation_scoring_seconds_by_block"] = {}
        for block in driver.BLOCKS:
            stage = time.perf_counter()
            np.save(folders[block]/"T_selected.npy",zt[block],allow_pickle=False)
            result = ctx.score(selections[block],selected[block],zt[block],folders[block])
            result["T_buried_exposed_activation_counts"] = driver.paired_activation_counts(ctx,zt[block])
            result.update(index=index,seed=20261118000+index,model="650m",block_one_based=block,
                descriptive_block_sensitivity=(block == 17),completed_utc=utc())
            driver.write(folders[block]/"RESULT.json",result)
            timings["confirmation_scoring_seconds_by_block"][str(block)] = time.perf_counter()-stage
        emit("PART1_COMPLETE",index=index)
        stage = time.perf_counter()
        direction_folder = folders[16]/"direction"
        direction_folder.mkdir()
        technical = direction.run_direction(net,vocab,ctx,zd[16],axes[16],16,propensity,direction_folder,
            site_batch=8,axis_batch=16)
        timings["direction_total_seconds"] = time.perf_counter()-stage
        timings["direction_substages"] = technical
        del zd,zt,selected
        required = [folder/"MODEL_IDENTITY.json",folder/"D_BOTH_BLOCKS_FROZEN.json"]
        for block in driver.BLOCKS:
            required += [folders[block]/name for name in ("D_axis_losses.npy","D_selected.npy",
                "sequence_model.npz","SELECTION_LOCK.json","T_selected.npy","RESULT.json",
                "group_errors.tsv","buried_exposed_group_errors.tsv")]
        required += [direction_folder/"DIRECTION_SEALED.json",direction_folder/"direction_vectors_SEALED.npz"]
        if any(not p.is_file() or p.stat().st_size == 0 for p in required):
            raise ValueError("A required scientific output is missing or empty")
        file_hashes = {str(p.relative_to(base)):driver.sha(p) for p in required}
    except Exception as error:
        driver.write(folder/"ERROR.json",{"utc":utc(),"index":index,"attempt":attempt,
            "type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc(),
            "same_seed_retry_requires_diagnosis":True})
        raise
    finally:
        if net is not None:
            del net
        gc.collect()
        torch.cuda.empty_cache()
    timings["total_seconds"] = time.perf_counter()-started
    receipt = {"status":"COMPLETE","index":index,"seed":20261118000+index,"attempt":attempt,
        "started_utc":started_utc,"completed_utc":utc(),"result_folder":str(folder),
        "blocks":[16,17],"direction_block":16,"timing":timings,"files":file_hashes,
        "scientific_values_disclosed":False,"runner_sha256":driver.sha(__file__)}
    driver.write(base/"COMPLETE.json",receipt)
    emit("NETWORK_COMPLETE",index=index,attempt=attempt,seconds=timings["total_seconds"])
    return receipt


def boundary_jobs(deadline):
    manifest = HERE/"GPU_BOUNDARY_JOBS.json"
    if not manifest.exists():
        return True
    jobs = driver.read(manifest)["jobs"]
    status_path = HERE/"GPU_BOUNDARY_STATUS.json"
    statuses = driver.read(status_path) if status_path.exists() else {"jobs":{}}
    for job in jobs:
        job_id = job["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+",job_id):
            raise ValueError("Unsafe GPU job identifier")
        if not job.get("enabled",True) or statuses["jobs"].get(job_id,{}).get("status") == "COMPLETE":
            continue
        if not deadline_check(deadline,job["estimated_seconds"]):
            progress("STOP_DEADLINE_BEFORE_BOUNDARY_JOB",next_job=job_id)
            return False
        argv = list(job["argv"])
        scripts = [Path(part).resolve() for part in argv if isinstance(part,str) and part.endswith(".py")]
        if not scripts or any(not p.is_relative_to(HERE) for p in scripts):
            raise ValueError("Boundary job must invoke only this extension's Python entry point")
        previous = statuses["jobs"].get(job_id,{})
        attempt = int(previous.get("attempt",0))+1
        if attempt > 3:
            raise ValueError("Boundary job exceeded three attempts")
        start = time.perf_counter()
        statuses["jobs"][job_id] = {"status":"RUNNING","started_utc":utc(),"attempt":attempt,
            "scripts_sha256":{str(p):driver.sha(p) for p in scripts},"PID_owner":os.getpid()}
        driver.write(status_path,statuses)
        emit("BOUNDARY_JOB_START",job_id=job_id,attempt=attempt)
        progress("BOUNDARY_JOB_RUNNING",active_job=job_id)
        log_path = HERE/"technical_gpu_logs"/f"{job_id}_attempt_{attempt:03d}.log"
        log_path.parent.mkdir(exist_ok=True)
        with log_path.open("w",encoding="utf8") as log:
            result = subprocess.run(argv,cwd=str(HERE),stdout=log,stderr=subprocess.STDOUT,text=True)
        seconds = time.perf_counter()-start
        state = "COMPLETE" if result.returncode == 0 else "ERROR"
        statuses["jobs"][job_id].update(status=state,exit_code=result.returncode,
            completed_utc=utc(),seconds=seconds,technical_log=str(log_path))
        driver.write(status_path,statuses)
        emit("BOUNDARY_JOB_"+state,job_id=job_id,seconds=seconds)
        if result.returncode != 0:
            raise RuntimeError("Boundary GPU job failed; diagnose technical log before same-job retry")
    return True


def main():
    gate,propensity = driver.gates(True,True)
    go = driver.read(HERE/"MAIN_GO.json")
    if (go.get("status") != "AUTHORIZED_MAIN" or go["K"] != 199 or
        go["frozen_protocol_sha256"] != gate["freeze"]["protocol_sha256"] or
        go["freeze_receipt_sha256"] != gate["freeze_sha256"]):
        raise ValueError("Main author authorization does not match the frozen protocol")
    deadline = datetime.fromisoformat(go["active_deadline_utc"].replace("Z","+00:00"))
    pilot = HERE/"sealed"/"timing_650m_init001"
    if driver.read(pilot/"COMPLETE.json")["status"] != "COMPLETE_SEALED_TIMING_PILOT":
        raise ValueError("The fixed init001 pilot is not complete")
    lock = HERE/"MAIN_GPU.lock"
    if lock.exists():
        prior = driver.read(lock)
        if alive(prior["PID"]):
            raise ValueError("Another GPU queue owner is active")
        lock.replace(HERE/f"MAIN_GPU_stale_{prior['PID']}_{time.time_ns()}.lock")
    with lock.open("x",encoding="utf8") as f:
        json.dump({"PID":os.getpid(),"started_utc":utc()},f)
    driver.write(HERE/"MAIN_RUNNER_RECEIPT.json",{"started_utc":utc(),"PID":os.getpid(),
        "runner_sha256":driver.sha(__file__),"MAIN_GO_sha256":driver.sha(HERE/"MAIN_GO.json"),
        "freeze_sha256":gate["freeze_sha256"],"frozen_sources":gate["frozen_python_sources_sha256"],
        "active_deadline_utc":go["active_deadline_utc"],"single_GPU_owner":True,
        "pilot_reused_folder":str(pilot),"automatic_error_retry":False})
    try:
        reference = driver.load_module("main_archived_context",driver.REFERENCE)
        direction = driver.load_module("main_direction_readout",HERE/"direction_readout.py")
        torch = reference.torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        emit("SHARED_CONTEXT_PREPARATION")
        ctx = reference.Context()
        pilot_seconds = driver.read(HERE/"TIMING_650M.json")["timing"]["complete_pilot_wall_seconds"]
        for index in range(2,200):
            if (HERE/"sealed"/"main_650m"/f"init_{index:03d}"/"COMPLETE.json").exists():
                continue
            if index > 2 and not boundary_jobs(deadline):
                return
            records = completed_records()
            estimate = float(np.median([r["timing"]["total_seconds"] for r in records[-10:]])) if records else pilot_seconds
            if not deadline_check(deadline,estimate):
                progress("STOP_DEADLINE_BEFORE_NETWORK",next_index=index,estimate_seconds=estimate)
                emit("STOP_DEADLINE",next_index=index,estimate_seconds=estimate)
                return
            progress("RUNNING",current_index=index,estimate_seconds=estimate)
            run_network(reference,direction,ctx,propensity,index)
            progress("NETWORK_BOUNDARY",last_completed_index=index)
        if not boundary_jobs(deadline):
            return
        progress("COMPLETE_650M_AND_REGISTERED_GPU_JOBS")
        emit("COMPLETE_650M_AND_REGISTERED_GPU_JOBS",completed_650m=199)
    except Exception as error:
        progress("STOP_ERROR",error_type=type(error).__name__,message=str(error))
        emit("STOP_ERROR",error_type=type(error).__name__,message=str(error))
        raise
    finally:
        if lock.exists() and driver.read(lock)["PID"] == os.getpid():
            lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
