from parser.service import ParserService
import yaml


def main():
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    service = ParserService(config)
    prompt = "Assign tasks TA and TB to Alice (cap1) and Bob (cap2). Costs: TA Alice 5 Bob 3, TB Alice 2 Bob 4. All tasks must be assigned."
    spec = service.parse_problem_spec(prompt)
    print("Parsed spec:")
    print(spec.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
