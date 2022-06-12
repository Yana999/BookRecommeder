import numpy as np

from bookRecommender.predict import make_prediction
from config.core import config
from pipeline import book_pipe
from processing.data_manager import load_dataset, save_pipeline


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.app_config.data_file)

    # fit model
    book_pipe.fit(data)

    # persist trained model
    save_pipeline(pipeline_to_persist=book_pipe)


if __name__ == "__main__":
    run_training()
    data = {'userID': [276725],
            'ISBN': ['034545104X'],
            'bookRating': [0],
            'bookTitle': ['Flesh Tones: A Novel'],
            'bookAuthor': ['M. J. Rose'],
            'yearOfPublication': [2002],
            'publisher': ['Ballantine Books'],
            'imageUrlS': ['http://images.amazon.com/images/P/034545104X.0...	'],
            'imageUrlM': ['http://images.amazon.com/images/P/034545104X.0...	'],
            'imageUrlL': ['http://images.amazon.com/images/P/034545104X.0...	'],
            'Location': ['tyler, texas, usa'],
            'Age': np.nan}
    print('trained')
    print(make_prediction('16 Lighthouse Road'))
