"""Hash stored tensor byte ranges, without loading a network or tensor array."""
import hashlib
import json
from pathlib import Path
import struct

OUT = Path(__file__).resolve().parent


def main():
    identity = json.loads((OUT / "MODEL_IDENTITY.json").read_text())
    weight = Path(identity["snapshot"]) / "model.safetensors"
    tensors = {}
    with weight.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
        base = size + 8
        for name, record in sorted(header.items()):
            if name == "__metadata__":
                continue
            begin, end = record["data_offsets"]
            stream.seek(base + begin)
            digest, remaining = hashlib.sha256(), end - begin
            while remaining:
                block = stream.read(min(remaining, 1024 * 1024))
                if not block:
                    raise ValueError("Truncated safetensors data")
                digest.update(block)
                remaining -= len(block)
            tensors[name] = {"shape": record["shape"], "dtype": record["dtype"],
                             "sha256": digest.hexdigest(), "data_offsets": [begin, end]}
    result = {"weight_path": str(weight), "weight_sha256": identity["weight_sha256"],
              "source": "Raw contiguous tensor data ranges in the already-verified source safetensors; not a second model load",
              "tensors": tensors,
              "loaded_CPU_model_state_sha256": identity["state_sha256"],
              "loaded_CPU_embedding_decoder_tied": identity["embedding_lm_head_tied"],
              "explicit_aliases": {"lm_head.decoder.weight": "esm.embeddings.word_embeddings.weight"},
              "bias_note": "Archived EsmLMHead has a standalone lm_head.bias; decoder is bias=False. Do not invent an archived decoder.bias tensor."}
    path = OUT / "PARAMETER_BYTES.json"
    temp = path.with_suffix(".json.partial")
    temp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    print(json.dumps({"status": "COMPLETE", "stored_tensors": len(tensors), "path": str(path)}))


if __name__ == "__main__":
    main()
