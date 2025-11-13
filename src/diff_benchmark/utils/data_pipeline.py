from diff_benchmark.preprocessing.wrapper_brain_data import (  # DefaultWandPipeline,
    DefaultHcpPipeline,
    ImageHcpPipeline,
    LcotEmbedHcpPipeline,
)


def get_data_pipeline(data_type, config):
    if data_type == "lcot_embed":
        print("Using LCOT Embeddings Pipeline")
        brain_preparator = LcotEmbedHcpPipeline(config)
    elif data_type == "images":
        print("Using Image Pipeline")
        brain_preparator = ImageHcpPipeline(config)
    elif data_type == "array":
        print("Using Default Array Pipeline")
        brain_preparator = DefaultHcpPipeline(config)

    return brain_preparator
