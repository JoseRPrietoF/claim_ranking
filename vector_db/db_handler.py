import lancedb
import pickle as pkl
from lancedb.pydantic import LanceModel

def load_pickle(path:str):
    """
    Load a pickle file and process the DataFrame.

    This function loads a DataFrame from a pickle file, renames specific columns,
    selects a subset of columns, removes duplicate rows based on the 'reviewed_claim'
    column, and fills any missing values with an empty string.

    Parameters:
    path (str): The file path to the pickle file.

    Returns:
    pandas.DataFrame: The processed DataFrame.
    """
    with open(path, 'rb') as file:
        df = pkl.load(file)
    columns = ['reviewed_claim',
       'url', 'title', 'text', 'summary', 'meta_description',
       'kb_keywords', 'meta_keywords', 'cr_country', 'meta_lang', 'cr_image',
       'meta_image', 'movies', 'domain', 'cm_authors', 'cr_author_name',
       'cr_author_url', 'cr_item_reviewed_text']
    change_cols = { 'reviewed claim': 'reviewed_claim',}
    df.rename(columns=change_cols, inplace=True)
    df = df[columns]
    df = df.drop_duplicates(subset='reviewed_claim').fillna('')

    return df

def search_top_k(query_vector:str, k=10):
    """
    Searches for the top k closest matches to the query_vector in the given table.

    Args:
        tbl (Table): The table to search within.
        query_vector (string): The string to search for.
        k (int, optional): The number of top results to return. Defaults to 10.

    Returns:
        list: A list of the top k closest matches, each containing the "reviewed_claim" field.
    """
    results = tbl.search(query_vector, vector_column_name="reviewed_claim").limit(k).select(["reviewed_claim"]).to_list()
    return [d["reviewed_claim"] for d in results]

class Content(LanceModel):
    reviewed_claim: str
    url: str
    title: str
    text: str
    summary: str
    meta_description: str
    kb_keywords: str
    meta_keywords: str
    cr_country: str
    meta_lang: str
    cr_image: str
    meta_image: str
    movies: str
    domain: str
    cm_authors: str
    cr_author_name: str
    cr_author_url: str
    cr_item_reviewed_text: str


# Initialize LanceDB connection and collection
DATA_TO_LOAD = "data/claims_te.pkl"
DB_PATH = "data/vector_db_lancedb"
COLLECTION_NAME = "claims"
db = lancedb.connect(DB_PATH)
if COLLECTION_NAME in db.table_names():
    tbl = db.open_table(COLLECTION_NAME)
else:
    print("Creating database with LanceDB")
    df = load_pickle(DATA_TO_LOAD)
    tbl = db.create_table(COLLECTION_NAME, df, schema=Content)
    tbl.create_fts_index("reviewed_claim", use_tantivy=False)
    print(f"Table {COLLECTION_NAME} created with {tbl.count_rows()} rows")