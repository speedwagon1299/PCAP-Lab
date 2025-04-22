import requests
from bs4 import BeautifulSoup

def fetch_arxiv_bibtex(arxiv_id):
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, features="xml")

    entry = soup.find("entry")
    if entry is None:
        return None

    title = entry.title.text.replace('\n', ' ').strip()
    year = entry.published.text[:4]
    authors = [a.find("name").text for a in entry.find_all("author")]
    author_string = " and ".join(authors)

    return f"""@article{{arxiv{arxiv_id.replace('.', '')},
  author = {{{author_string}}},
  title = {{{title}}},
  journal = {{arXiv preprint arXiv:{arxiv_id}}},
  year = {{{year}}},
  note = {{[Online]. Available: https://arxiv.org/abs/{arxiv_id}}}
}}"""

# Example usage:
arxiv_ids = [
    "2309.14859"
]

for aid in arxiv_ids:
    bib = fetch_arxiv_bibtex(aid)
    if bib:
        print(bib + "\n")
