from bookRecommender.config.core import config
from bookRecommender.processing import features


def test_null_variable_transformer(sample_input_data):
    # Arrange
    transformer = features.NullVariableTransformer(
                config.model_config.subject,
                config.model_config.user,
                config.model_config.rating_variable
            )

    # Apply
    subject = transformer.fit_transform(sample_input_data)

    # Assert
    assert not subject[config.model_config.subject].isna().any()
    assert not subject[config.model_config.subject].is_unique
    assert not subject[config.model_config.user].is_unique


def test_add_variable_transformer(sample_input_data):
    # Arrange
    transformer = features.AddVariableTransformer(
                config.model_config.subject,
                config.model_config.user,
                config.model_config.rating_variable,
                config.model_config.new_feature
            )

    # Apply
    subject = transformer.fit_transform(sample_input_data)

    # Assert
    assert config.model_config.new_feature in subject.columns
    assert not subject[config.model_config.new_feature].isna().any()


def test_restrict_variable_transformer(sample_input_data):
    # Arrange
    transformerAdd = features.AddVariableTransformer(
        config.model_config.subject,
        config.model_config.user,
        config.model_config.rating_variable,
        config.model_config.new_feature
    )
    transformerRestr = features.RestrictVariablesTransformer(
        config.model_config.location,
        config.model_config.popularity_threshold,
        config.model_config.new_feature,
        config.model_config.specific_location
    )

    # Apply
    subject = transformerRestr.fit_transform(transformerAdd.fit_transform(sample_input_data))

    # Assert
    assert (subject[config.model_config.new_feature] >= config.model_config.popularity_threshold).all()
    assert (subject[config.model_config.location].str.contains(config.model_config.specific_location)).all()