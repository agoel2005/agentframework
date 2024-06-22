import json


def check_interview_score(data_path: str) -> str:
    """
    Check the interview score of a candidate.
    """
    print("Testing")
    with open(data_path, "r") as file:
        data = json.load(file)

    round_1_excitment = data["R1: Excitement"]
    round_2_interview_1 = data["SWE R2 I1: Overall"]
    round_2_interview_2 = data["SWE R2 I2: Overall"]
    interview_score = (round_1_excitment + round_2_interview_1 + round_2_interview_2) / 3

    if interview_score > 4:
        return "Good fit"
    else:
        return "Not a good fit"

def main():
    data_path = "/Users/eliolcott/Desktop/Code/Hackathons/MIT_Innovation_HQ/agentframework/tests/test_1/dataset.csv"
    result = check_interview_score(data_path)
    print(result)

if __name__ == "__main__":
    main()
