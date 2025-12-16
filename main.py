from src.runner.base import Runner

def main():
    runner = Runner(csv_path="data/vit-human-alignment-framework-test.csv")
    runner.execute()

if __name__ == "__main__":
    main()
