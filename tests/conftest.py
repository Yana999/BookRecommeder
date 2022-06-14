import pytest

from bookRecommender.config.core import config
from bookRecommender.processing.data_manager import load_dataset


@pytest.fixture()
def sample_input_data():
    return load_dataset(file_name_book=config.app_config.data_file_book,
                        file_name_user=config.app_config.data_file_user,
                        file_name_rating=config.app_config.data_file_ratings)


@pytest.fixture()
def prediction_data():
    input_data = load_dataset(file_name_book=config.app_config.data_file_book,
                              file_name_user=config.app_config.data_file_user,
                              file_name_rating=config.app_config.data_file_ratings)
    return input_data['bookTitle'][0]
