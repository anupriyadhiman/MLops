from classification_model.config.core import config
from classification_model.processing.features import ExtractFirstLetterTransformer


def test_temporal_variable_transformer(sample_input_data):
    # Given
    transformer = ExtractFirstLetterTransformer(
        variables=config.model_config.temporal_vars,  
        reference_variable=config.model_config.ref_var,
    )
    assert sample_input_data["cabin"].iat[6] == "G6"

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert subject["cabin"].iat[6] == 49
