from bookRecommender.config.core import config
from bookRecommender.predict import make_prediction


def test_make_prediction(prediction_data):
    # Arrange
    expected_result_len = config.model_config.num_neighbors

    # Apply
    result = make_prediction(input_data=prediction_data)

    # Assert
    assert isinstance(result, list)
    assert isinstance(result[0], str)
    assert len(result) == expected_result_len