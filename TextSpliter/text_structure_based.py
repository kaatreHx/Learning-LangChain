from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
    In today’s rapidly evolving digital landscape, nonprofits are being asked to do more with less while facing urgent challenges and increased demand for services. Artificial Intelligence (AI) offers new opportunities to increase efficiency, improve programs, and drive mission-critical outcomes. There is a wealth of AI resources and learning products, but very few are tailored for nonprofit professionals or are created by nonprofit professionals.
    That is why NetHope and Microsoft Elevate have collaborated to launch a practical, nonprofit-focused AI Skills program. Unlocking AI for Nonprofits is a free, CPD-certified course series that helps nonprofit teams build the skills and confidence they need to use AI effectively, safely, and in service of their mission.
    Whether you’re just starting to explore AI or leading your organization’s digital transformation, this series meets you where you are, with practical tools, real nonprofit use cases, and a values-first approach to adoption.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separators=''
)

chunks = splitter.split_text(text)

print(chunks)