class VertexPositionModel:
    data_type = "lcot_embed"
    def __init__(self, transform_fn=None, **kwargs):
        """
        Parameters
        ----------
        transform_fn : callable or None
            A function applied to features["attenuations"] during fit/predict.
            If None, a default identity transform is used.
        """
        self.transform_fn = transform_fn or (lambda x: x)
        
    def fit(self, dataloader):
        """
        Dummy fit function.
        Iterates over the dataloader and processes:
        features -> features["attenuations"] -> transform.
        """
        print("Starting fit...")

        for batch_idx, batch in enumerate(dataloader):

            # Expecting batch = (features, targets, _)
            features, _, _ = batch
           # Extract attenuations
            atten = features["attenuations"]
            power = features["power"]
            positions = features["coords"]
            vertices = features["vertices"]

            # Transform attenuations (dummy transformation)
            atten_transformed = self.transform_fn(atten)

        print("Fit complete.")

    # -----------------------------------------------------------
    # Predict
    # -----------------------------------------------------------
    def predict(self, dataloader):
        """
        Dummy predict function.
        Returns the transformed attenuations for each batch.
        """
        
        outputs = []

        for batch_idx, batch in enumerate(dataloader):

            features, _, _ = batch

            atten = features["attenuations"]
            power = features["power"]
            positions = features["coords"]

            # Apply the same transformation used during fit
            atten_transformed = self.transform_fn(atten)

            outputs.append(atten_transformed)

        return outputs
