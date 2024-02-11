from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(data, target_column):
    """Предобработка данных: разделение на признаки и цель, нормализация."""
    X = data.drop([target_column, 'DATE'], axis=1)
    
    # Замена NaN значений в 'RAIN' на False (0) или на моду
    data[target_column] = data[target_column].fillna(0)  # или data[target_column].fillna(data[target_column].mode()[0])
    y = data[target_column].astype('int')
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Нормализация данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

