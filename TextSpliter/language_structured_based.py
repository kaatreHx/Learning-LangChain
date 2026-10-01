from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

code = """
import math

def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    
    if n == 2:
        return True
    
    if n % 2 == 0:
        return False
    
    max_divisor = int(math.isqrt(n))
    for i in range(3, max_divisor + 1, 2):
        if n % i == 0:
            return False 
            
    return True  

num = int(input("Enter a number to check: "))

if is_prime(num):
    print(f"{num} is a prime number.")
else:
    print(f"{num} is not a prime number.")

"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=0
)

chunks = splitter.split_text(code)

print(len(chunks))
print(chunks[1])