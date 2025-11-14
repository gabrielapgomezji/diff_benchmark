from diff_benchmark.preprocessing.wrapper_brain_data import (
    DefaultHcpPipeline,
    ImageHcpPipeline,
    LcotEmbedHcpPipeline,
)


def get_data_pipeline(data_type, config):
    """Factory function to get the appropriate data pipeline based on data_type."""
    if data_type == "lcot_embed":
        print("Using LCOT Embeddings Pipeline")
        brain_preparator = LcotEmbedHcpPipeline(config)
    elif data_type == "images":
        print("Using Image Pipeline")
        brain_preparator = ImageHcpPipeline(config)
    elif data_type == "array":
        print("Using Default Array Pipeline")
        brain_preparator = DefaultHcpPipeline(config)
    else:
        raise ValueError(
            f"Unknown data_type '{data_type}'. Must be one of ['lcot_embed', 'images', 'array']."
        )

    return brain_preparator
