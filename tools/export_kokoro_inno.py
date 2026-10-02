from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

INNO_VERSION = "0.2.0"
INNO_COMMIT = "892ef184bc932aa3ff9d72c1509d5b81ff6941e6"
GRAPH_INPUTS = ("fbank", "tilt", "head_stats", "blend_weights", "head_enabled")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_inno_voicepack(
    checkpoint: Path, destination: Path, metadata_destination: Path | None = None
) -> tuple[Path, Path]:
    """Export a local upstream v0.2.0 checkpoint as one stock-pack ONNX graph."""
    os.environ["HF_HUB_OFFLINE"] = "1"

    import numpy as np
    import onnx
    import torch
    import torch.nn.functional as functional
    from inno_kokoro.enroll import Tuner

    tuner = Tuner(str(checkpoint), device="cpu")
    if tuner.version != INNO_VERSION:
        raise ValueError(f"Expected Inno weights v{INNO_VERSION}, found v{tuner.version}")
    if not tuner.head:
        raise ValueError("The selected Inno checkpoint has no v0.2.0 prosody head")

    class VoicepackGraph(torch.nn.Module):
        def __init__(self, source_tuner):
            super().__init__()
            self.encoder = source_tuner.encoder
            self.style = source_tuner.style
            self.register_buffer("tilt_direction", source_tuner.tilt_dir.detach().float())
            self.register_buffer("blend_rows", source_tuner.rows.detach().float())
            self.register_buffer("head_weight", source_tuner.head["W"].detach().float())
            self.register_buffer("head_mean", source_tuner.head["mu"].detach().float())
            self.register_buffer("head_std", source_tuner.head["sd"].detach().float())

        def forward(self, fbank, tilt, head_stats, blend_weights, head_enabled):
            backbone = self.encoder.backbone
            features = fbank.permute(0, 2, 1).unsqueeze(1)
            hidden = functional.relu(backbone.bn1(backbone.conv1(features)))
            for index in range(4):
                hidden = getattr(backbone, f"layer{index + 1}")(hidden)
            pooled = torch.cat(
                [
                    hidden.mean(-1).flatten(1),
                    torch.sqrt(hidden.var(-1) + 1.0e-7).flatten(1),
                ],
                dim=1,
            )
            embedding = functional.normalize(self.encoder.proj(backbone.seg_1(pooled)), dim=-1)
            style = self.style(embedding) + self.tilt_direction.unsqueeze(0) * tilt.reshape(1, 1)
            acoustic = style[:, None, :128].expand(-1, 510, -1)

            weighted_rows = self.blend_rows.unsqueeze(0) * blend_weights[:, :, None, None]
            predictor = weighted_rows.sum(dim=1)
            normalized_head = (head_stats - self.head_mean) / self.head_std
            head_input = torch.cat([normalized_head, torch.ones_like(head_enabled[:, None])], dim=1)
            delta = head_input @ self.head_weight
            predictor = predictor + head_enabled[:, None, None] * delta[:, None, :]
            return torch.cat([acoustic, predictor], dim=-1).permute(1, 0, 2)

    model = VoicepackGraph(tuner).eval().requires_grad_(False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_destination = metadata_destination or destination.with_name("inno_tuner.npz")
    if destination.resolve() == metadata_destination.resolve():
        raise ValueError("Graph and tuner metadata destinations must be different files")
    if metadata_destination.suffix.lower() != ".npz":
        raise ValueError("Tuner metadata destination must use the .npz extension")
    metadata_destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = _sha256(checkpoint)
    np.savez_compressed(
        metadata_destination,
        blend_names=np.asarray(tuner.names, dtype=np.str_),
        blend_stats=np.asarray(tuner.stats, dtype=np.float32),
        blend_grades=np.asarray(tuner.grades, dtype=np.float32),
        blend_scale=np.asarray(tuner.scale),
        blend_gate=np.asarray(tuner.gate, dtype=np.float32),
        grade_pen=np.asarray(tuner.grade_pen, dtype=np.float32),
        tilt_mean=np.asarray(tuner.tilt_norm[0], dtype=np.float32),
        tilt_sd=np.asarray(tuner.tilt_norm[1], dtype=np.float32),
        version=np.asarray(INNO_VERSION),
        upstream_commit=np.asarray(INNO_COMMIT),
        checkpoint_sha256=np.asarray(checkpoint_sha256),
    )
    metadata_sha256 = _sha256(metadata_destination)
    example_inputs = (
        torch.zeros((1, 300, 80), dtype=torch.float32),
        torch.zeros((1,), dtype=torch.float32),
        torch.zeros((1, 2), dtype=torch.float32),
        torch.full((1, len(tuner.names)), 1.0 / len(tuner.names), dtype=torch.float32),
        torch.ones((1,), dtype=torch.float32),
    )
    with torch.no_grad():
        torch.onnx.export(
            model,
            example_inputs,
            str(destination),
            input_names=list(GRAPH_INPUTS),
            output_names=["voicepack"],
            dynamic_axes={"fbank": {1: "frames"}},
            opset_version=17,
            dynamo=False,
        )

    graph = onnx.load(str(destination))
    onnx.checker.check_model(graph)
    properties = {
        "inno_upstream_version": INNO_VERSION,
        "inno_upstream_commit": INNO_COMMIT,
        "inno_checkpoint_sha256": checkpoint_sha256,
        "inno_metadata_sha256": metadata_sha256,
        "inno_blend_names": ",".join(tuner.names),
        "inno_license": "Apache-2.0; embedded speaker encoder weights CC BY-SA 3.0",
    }
    for key, value in properties.items():
        item = graph.metadata_props.add()
        item.key = key
        item.value = value
    onnx.save(graph, str(destination))
    return destination, metadata_destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", type=Path, required=True, help="local Inno v0.2.0 model.safetensors"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="destination inno_voicepack.onnx"
    )
    parser.add_argument("--metadata-output", type=Path, help="destination tuner metadata NPZ")
    arguments = parser.parse_args()
    for path in export_inno_voicepack(
        arguments.weights, arguments.output, arguments.metadata_output
    ):
        print(path)


if __name__ == "__main__":
    main()
