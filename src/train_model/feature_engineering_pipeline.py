
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_preprocessor():
    """
    Creación del preprocesador para los datos de entrada.
    Este preprocesador maneja columnas ordinales, nominales y numéricas.
    """
    # Columnas que serán codificadas ordinalmente
    ordinal_cols = ['Class_ X_Percentage', 'Class_XII_Percentage']

    # Columnas que serán tratadas como numéricas (ahora solo 'time')
    #numeric_cols = ['time']

    # Columnas que recibirán One-Hot Encoding
    nominal_cols = [
        'Gender', 'Caste', 'coaching', 'Class_ten_education',
        'twelve_education', 'medium', 'Father_occupation', 'Mother_occupation',
        'time'
    ]

    percentage_order_list = ['Average', 'Good', 'Vg', 'Excellent']

    # Pipeline para datos ordinales categóricos
    ordinal_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=[percentage_order_list] * len(ordinal_cols)))
    ])

    # Pipeline para datos nominales (categóricos sin orden)
    nominal_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
    ])

    # Pipeline para la única columna numérica ('time')
    # numeric_pipe = Pipeline(steps=[
    #     ('imputer', SimpleImputer(strategy='median'))
    # ])

    # Unir todos los pipelines en el ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('ordinal', ordinal_pipe, ordinal_cols),
        ('nominal', nominal_pipe, nominal_cols),
        # ('numeric', numeric_pipe, numeric_cols)
    ], remainder='passthrough')


    return preprocessor