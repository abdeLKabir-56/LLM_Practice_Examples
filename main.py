from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


llm = OllamaLLM(
    model="llama3.2"
)

data = """
Elon Musk (prononcé en anglais : né le 28 juin 1971 à Pretoria (Afrique du Sud), est un entrepreneur, homme d'affaires et homme politique sud-africain naturalisé canadien et américain. Il est depuis 2024 la personne la plus riche du monde et depuis 2026, la première personne de l'histoire à avoir dépassé un seuil de fortune de plus de mille milliards de dollars[1],[2].

Elon Musk commence sa carrière en affaires comme cofondateur de la société de logiciels Zip2 avec son frère, Kimbal Musk. La start-up est acquise par Compaq pour 307 millions de dollars en 1999. La même année, Musk cofonde la banque en ligne X.com, qui fusionne avec Confinity en 2000 pour former PayPal. eBay rachète PayPal en 2002 pour 1,5 milliard de dollars.

En 2002, Musk fonde SpaceX, un fabricant aérospatial et une société de services de transport spatial, et en est le président-directeur général. En 2004, il investit 6,5 millions de dollars dans le constructeur de véhicules électriques Tesla, et en devient le PDG en 2008. En 2006, il participe à la création de SolarCity, une société d'énergie solaire qui est ensuite acquise par Tesla et devient Tesla Energy. En 2015, il cofonde et devient coprésident d'OpenAI, une association de recherche promouvant l'intelligence artificielle amicale, qu'il quitte en 2018. En 2016, il fonde The Boring Company, société de construction de tunnels (Hyperloop), et Neuralink, société de neurotechnologie. En 2022, il devient le propriétaire de Twitter par un achat à 44 milliards de dollars, qu'il renomme « X » l'année suivante. En 2023, il fonde la société xAI dans le domaine de l'intelligence artificielle.

En janvier 2021, selon Bloomberg, Elon Musk devient, à 49 ans, l'individu le plus riche du monde avec une fortune estimée à 188,5 milliards de dollars. Au cours du dernier trimestre 2025, elle atteint presque 730 milliards au 31 décembre 2025 à la suite de la revalorisation de ses participations.

À partir de la fin des années 2010, ses actions et déclarations, dont certaines relèvent de la désinformation et des théories du complot, sont régulièrement médiatisées.

Lors du premier mandat présidentiel de Donald Trump, Elon Musk devient son conseiller officieux, jusqu'à ce qu'un différend les oppose à propos de l'accord de Paris sur le climat. Alors que ses positions continuent de s'orienter vers l'extrême droite, Musk devient l'un des plus importants soutiens de l'ancien président en vue de sa réélection et s'implique dans sa campagne de 2024, dont il est le deuxième plus gros contributeur financier.

Après la victoire de Donald Trump, il est nommé haut conseiller et prend la tête du département de l'Efficacité gouvernementale, instance temporaire qui procède à des coupes drastiques dans de nombreux programmes et agences gouvernementaux et licencie des dizaines de milliers de salariés fédéraux. Au bout de quatre mois, il quitte la direction du département et entre dans un violent conflit avec Donald Trump au sujet de sa politique fiscale. Chantre du natalisme, promoteur du nationalisme blanc et du concept de remigration[3], il étend par ailleurs son influence sur les pays européens, apportant publiquement son soutien à des partis d'extrême droite européens.
    """

template = """
You are a helpful assistant that summarizes the following text in French {data}. 
- Please provide a concise summary of the main points and key information contained in the text.
"""

prompt = ChatPromptTemplate.from_template(template = template)

chain = prompt | llm

llm_response = chain.invoke(input = {"data" : data})

def main():
    print("Hello from ai-agent-ollama-langchain!")
    print("LLM Response: \n" , llm_response)


if __name__ == "__main__":
    main()