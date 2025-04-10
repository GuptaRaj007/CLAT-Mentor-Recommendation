import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import random

# Set random seed for reproducibility
np.random.seed(42)

# Mock data generation
def generate_mock_data(num_aspirants=100, num_mentors=20):
    subjects = ['Legal Reasoning', 'Logical Reasoning', 'English', 'General Knowledge', 'Mathematics']
    colleges = ['NLU Delhi', 'NALSAR', 'NUJS', 'NLIU', 'GNLU']
    prep_levels = ['Beginner', 'Intermediate', 'Advanced']
    learning_styles = ['Visual', 'Auditory', 'Kinesthetic']
    
    aspirants = []
    for i in range(num_aspirants):
        aspirant = {
            'id': f'A{i+1:03d}',
            'preferred_subjects': random.sample(subjects, random.randint(2, 4)),
            'target_colleges': random.sample(colleges, random.randint(1, 3)),
            'preparation_level': random.choice(prep_levels),
            'learning_style': random.choice(learning_styles),
            'mock_test_score': random.randint(50, 200)
        }
        aspirants.append(aspirant)
    
    mentors = []
    for i in range(num_mentors):
        mentor = {
            'id': f'M{i+1:03d}',
            'specialized_subjects': random.sample(subjects, random.randint(2, 4)),
            'alma_mater': random.choice(colleges),
            'teaching_experience': random.randint(1, 5),
            'teaching_style': random.choice(learning_styles),
            'clat_score': random.randint(150, 200),
            'rating': round(random.uniform(3.5, 5), 2),
            'students_mentored': random.randint(5, 50)
        }
        mentors.append(mentor)
    
    return pd.DataFrame(aspirants), pd.DataFrame(mentors)

# Preprocessing
def preprocess_data(aspirants_df, mentors_df):
    aspirants = aspirants_df.copy()
    mentors = mentors_df.copy()

    # Keep original columns for later display
    mentors_meta = mentors[['id', 'alma_mater', 'specialized_subjects', 'teaching_style',
                            'clat_score', 'rating', 'students_mentored']].copy()

    all_subjects = list(set([s for subs in aspirants['preferred_subjects'] for s in subs]))
    all_colleges = list(set([c for cols in aspirants['target_colleges'] for c in cols]))

    for subject in all_subjects:
        aspirants[f'subject_{subject}'] = aspirants['preferred_subjects'].apply(lambda x: 1 if subject in x else 0)
        mentors[f'subject_{subject}'] = mentors['specialized_subjects'].apply(lambda x: 1 if subject in x else 0)

    for college in all_colleges:
        aspirants[f'college_{college}'] = aspirants['target_colleges'].apply(lambda x: 1 if college in x else 0)
        mentors[f'college_{college}'] = mentors['alma_mater'].apply(lambda x: 1 if college == x else 0)

    aspirants = pd.get_dummies(aspirants, columns=['preparation_level', 'learning_style'])
    mentors = pd.get_dummies(mentors, columns=['teaching_style'])

    # Add missing columns (alignment)
    all_feature_cols = set(aspirants.columns).union(set(mentors.columns))
    for col in all_feature_cols:
        if col.startswith(('subject_', 'college_', 'preparation_level_', 'learning_style_', 'teaching_style_')):
            if col not in aspirants.columns:
                aspirants[col] = 0
            if col not in mentors.columns:
                mentors[col] = 0

    # Normalize numerical features
    scaler = MinMaxScaler()
    aspirants['mock_test_score'] = scaler.fit_transform(aspirants[['mock_test_score']])
    mentors['clat_score'] = scaler.fit_transform(mentors[['clat_score']])
    mentors['rating'] = scaler.fit_transform(mentors[['rating']])

    # Define common features for similarity
    common_features = [col for col in aspirants.columns if col.startswith(('subject_', 'college_', 'preparation_level_', 'learning_style_'))]
    
    aspirants['score'] = aspirants['mock_test_score']
    mentors['score'] = mentors[['clat_score', 'rating']].mean(axis=1)

    common_features.append('score')

    # Merge back mentor metadata
    mentors = pd.concat([mentors_meta, mentors.drop(columns=['id'])], axis=1)

    return aspirants, mentors, common_features

# Recommendation
def recommend_mentors(aspirant_id, aspirants_df, mentors_df, common_features, n_recommendations=3):
    aspirant_row = aspirants_df[aspirants_df['id'] == aspirant_id].iloc[0]
    aspirant_vector = aspirant_row[common_features].values.reshape(1, -1)
    mentor_vectors = mentors_df[common_features].values
    
    similarities = cosine_similarity(aspirant_vector, mentor_vectors)[0]
    top_indices = similarities.argsort()[-n_recommendations:][::-1]
    
    recommendations = mentors_df.iloc[top_indices].copy()
    recommendations['similarity_score'] = similarities[top_indices]
    
    return recommendations[['id', 'alma_mater', 'specialized_subjects', 'teaching_style', 
                            'clat_score', 'rating', 'students_mentored', 'similarity_score']]

# Main
aspirants_df, mentors_df = generate_mock_data()
aspirants_processed, mentors_processed, features = preprocess_data(aspirants_df, mentors_df)

aspirant_id = 'A001'
recommendations = recommend_mentors(aspirant_id, aspirants_processed, mentors_processed, features)

print(f"\nTop 3 mentor recommendations for aspirant {aspirant_id}:\n")
print(recommendations)
