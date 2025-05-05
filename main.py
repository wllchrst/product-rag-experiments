'''Main python file for running the application'''
import pandas as pd
from interfaces import ProductInformation, Review, EmotionBasedInput, ConclusionBasedInput
from agents import EmotionBasedAgent, ConclusionAgent
from typing import List
from dataclasses import asdict

def get_example() -> List[Review]:
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

    return reviews

product_information: ProductInformation = ProductInformation(
    name="Sample Product",
    description="This is a sample product.",
    price=19.99,
    overall_rating=4.5
)

def main():
    reviews = get_example()
    conclusions = []
    for review_data in reviews[0:2]:
        ebi = EmotionBasedInput(
            product_information=product_information,
            reviews=review_data
        )

        agent = EmotionBasedAgent(config_key='emotion_based_config')
        result = agent.execute_task(data=asdict(ebi))
        conclusions.append(result)

    # print(f'total conclusions: {len(conclusions)}')
    cbi = ConclusionBasedInput(conclusions=conclusions)
    ca = ConclusionAgent(config_key='conclusion_based_config')
    final_answer = ca.execute_task(data=asdict(cbi))
    print(f'Final Answer: {final_answer}')
    
if __name__ == '__main__':
    main()