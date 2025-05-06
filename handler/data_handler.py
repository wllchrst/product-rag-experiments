import pandas as pd
from typing import List, Tuple
from interfaces import Review, ProductInformation 
class DataHandler:
    def get_dummy_data(self) -> Tuple[List[Review], ProductInformation]:
        df = pd.read_csv('./data/PRDECT-ID Dataset.csv')
        reviews = []

        df = df[['Customer Review', 'Emotion', 'Product Name', 'Customer Rating']]
        groups = df.groupby('Emotion')

        for _, group in groups:
            top_reviews = group.head(5)
            emotion_reviews = []

            for _, row in top_reviews.iterrows():
                r = Review(
                    review=row['Customer Review'],
                    rating=str(row['Customer Rating']),
                    emotion=row['Emotion']
                )
                emotion_reviews.append(r)

            reviews.append(emotion_reviews)
        
        dummy_product_information: ProductInformation = ProductInformation(
            name="Sample Product",
            description="This is a sample product.",
            price=19.99,
            overall_rating=4.5
        )

        return (reviews, dummy_product_information)