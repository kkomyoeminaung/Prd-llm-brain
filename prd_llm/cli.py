"""
Command-line interface for PRD-LLM
"""
import argparse

def main():
    parser = argparse.ArgumentParser(description="PRD-LLM CLI")
    parser.add_argument("command", choices=["chat", "generate", "serve", "info"])
    args = parser.parse_args()
    
    if args.command == "info":
        print("PRD-LLM v2.0.0 | Human-Like Brain Architecture")
    else:
        print(f"Running command: {args.command}")

if __name__ == "__main__":
    main()
