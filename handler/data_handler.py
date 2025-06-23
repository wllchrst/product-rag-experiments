import pandas as pd
from typing import List, Tuple
from interfaces import Review, ProductInformation, UserReview
class DataHandler:
    def __init__(self):
        self.product_data = pd.read_excel('./data/product_data.xlsx')
        self.reviews = pd.read_excel('./data/product_reviews.xlsx')
        self.user_reviews = pd.read_excel('./data/user_reviews.xlsx')

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
    
    def get_user_reviews(self, product_name: str) -> List[UserReview]:
        user_review = self.user_reviews[self.user_reviews['product_name'] == product_name]
        if user_review.empty:
            raise ValueError(f"No user reviews found for product: {product_name}")

        user_reviews: List[UserReview] = []
        for _, review_row in user_review.iterrows():
            r = UserReview(
                user_name=review_row['user_name'],
                product_name=review_row['product_name'],
                review=review_row['user_input']
            )
            user_reviews.append(r)

        return user_reviews
    
    def get_data(self, product_name: str) -> Tuple[List[Review], ProductInformation]:
        product_data = self.product_data[self.product_data['name'] == product_name].iloc[0]
        if product_data.empty:
            raise ValueError(f"No data found for product: {product_name}")
        
        reviews_df = self.reviews[self.reviews['product_name'] == product_name]
        if reviews_df.empty:
            raise ValueError(f"No reviews found for product: {product_name}")
        
        product_information: ProductInformation = ProductInformation(
            name=product_data['name'],
            description=product_data['description'],
            price=product_data['price'],
            overall_rating=product_data['overall_rating'],
            link=product_data['link']
        )

        reviews = []
        for _, review_row in reviews_df.iterrows():
            r = Review(
                review=review_row['review'],
                rating=str(review_row['rating']),
                emotion=''
            )

            reviews.append(r)

        return reviews, product_information