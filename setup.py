from setuptools import find_packages, setup


def get_requirements(file_path: str) -> list[str]:
    """Read dependencies from the requirements file, skipping '-e .'."""
    with open(file_path, encoding="utf-8") as f:
        requirements = [line.strip() for line in f]
    return [req for req in requirements if req and not req.startswith(("#", "-e"))]


setup(
    name="medical-chatbot",
    version="0.1.0",
    author="Dhanushka Gunasinghe",
    author_email="dhanushkapg@gmail.com",
    description="A medical chatbot built with LangChain, Pinecone and Flask",
    packages=find_packages(),
    install_requires=get_requirements("requirments.txt"),
    python_requires=">=3.10",
)
