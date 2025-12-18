from diff_benchmark.preprocessing.base_demographic_data import DemographicsPreprocessor

COLUMN_ALIASES = {
    "Subject": {
        "subject", "participant_id", "participant", "sub_id", "sub",
    },
    "Age": {
        "age", "age_in_yrs", "age_in_years", "age_years", "ageyrs",
    },
    "Gender": {
        "gender", "sex", "gender_text",
    },
}

class DefaultDemographicsPreprocessor(DemographicsPreprocessor):
    """
    DefaultDemographicsPreprocessor is a class that extends the DemographicsPreprocessor
    to preprocess demographic data specifically for a dataset.
    Methods:
        filter(target_columns: list[str]) -> None:
            Filters the DataFrame to include only the specified target columns,
            ensuring that "Subject" and "Gender" are included if available.
        categorical_to_numeric() -> None:
            Converts the "Gender" column from categorical values ("M", "F") to numeric
            values (1 for "M" and 0 for "F") if the column exists and is of object type.
        clean_df() -> None:
            Cleans the DataFrame by removing any rows with missing values.
    """

    def filter(self, target_columns: list[str]) -> None:
        if self.df.index.name and self.df.index.name.lower() in COLUMN_ALIASES["Subject"]:
            self.df = self.df.reset_index()
        self.df = self.df.rename(columns={
            c: canonical
            for canonical, aliases in COLUMN_ALIASES.items()
            for c in self.df.columns
            if c.lower() in aliases
        })
        # Always include "Subject" and "Gender" if available
        columns = ["Subject"] + target_columns
        if "Gender" not in columns and "Gender" in self.df.columns:
            columns.append("Gender")
        self.df = self.df.loc[:, [col for col in columns if col in self.df.columns]]

    def categorical_to_numeric(self) -> None:
        if "Gender" in self.df.columns and self.df["Gender"].dtype == object:
            self.df["Gender"] = (self.df["Gender"].astype(str)
                                .str.upper()
                                .map({"M": 1, "F": 0, "MALE": 1, "FEMALE": 0}))

    def clean_df(self) -> None:
        self.df = self.df.dropna()

