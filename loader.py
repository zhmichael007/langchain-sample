#csv
from langchain.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path='./mlb_teams_2012.csv')
data = loader.load()
print(data)

#pdf
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("./user_guide.pdf")
pages = loader.load_and_split()
print(pages[0])
print(pages[1])