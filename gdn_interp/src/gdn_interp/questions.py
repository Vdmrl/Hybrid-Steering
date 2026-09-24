import random


SIMPLE_QUESTIONS = (
    "What might happen if someone misses the last bus home?",
    "Describe an afternoon spent walking in the rain.",
    "How would you decide whether to take an umbrella?",
    "What could happen when two strangers find the same lost key?",
    "Describe a quiet morning in a nearly empty cafe.",
    "How might a person react to an unexpected knock at the door?",
    "What are some possible outcomes of changing plans at the last minute?",
    "Describe a child opening an unmarked box.",
    "How would you decide which route to take on a walk?",
    "What might happen when the lights go out during dinner?",
    "Describe a conversation between neighbors about a noisy evening.",
    "How might someone choose between staying home and going outside?",
    "What could explain a dog barking at an empty garden?",
    "Describe a person waiting alone at a train station.",
    "How would you respond if a friend arrived much later than expected?",
    "What might happen after someone finds a note under a chair?",
    "Describe a windy day in a small town.",
    "How could a group decide where to have lunch?",
    "What are some reasons a package might arrive late?",
    "Describe someone trying a new hobby for the first time.",
    "How might a person decide whether to trust a vague message?",
    "What could happen if a meeting is moved to an unfamiliar place?",
    "Describe an evening when the weather suddenly changes.",
    "How would you choose a gift for someone you barely know?",
    "What might be behind a closed door in an old building?",
    "Describe two friends taking different paths through a park.",
    "How could someone prepare for a day with uncertain weather?",
    "What are some possible reasons a room is unusually quiet?",
    "Describe a traveler arriving in a town after dark.",
    "How might a person react after making a small mistake?",
    "What could happen if a bicycle is left outside overnight?",
    "Describe a short visit to an unfamiliar shop.",
    "How would you decide whether to accept an unexpected invitation?",
    "What might happen when someone hears music from another room?",
    "Describe a family choosing what to do on a free afternoon.",
    "How could a person interpret footprints near their front door?",
    "What are some possible outcomes of planting seeds in a garden?",
    "Describe someone searching for a missing notebook.",
    "How might a team handle a plan that is not working well?",
    "What could happen during an unplanned stop on a road trip?",
    "Describe a person watching clouds gather over a field.",
    "How would you decide whether an unfamiliar path is safe to follow?",
    "What might explain a light turning on in an empty office?",
    "Describe a meal made from whatever is left in the kitchen.",
    "How could someone respond to a rumor about a local event?",
    "What are some possible endings to a disagreement between friends?",
    "Describe a morning when nothing goes according to plan.",
    "How might a person decide whether to wait or leave?",
    "What could happen after a small boat drifts away from the shore?",
    "Describe someone discovering an unfamiliar object in a coat pocket.",
)


def simple_questions(questions: int = 50, seed: int = 42) -> list[dict[str, str]]:
    if not 1 <= questions <= len(SIMPLE_QUESTIONS):
        raise ValueError(f"questions must be in [1, {len(SIMPLE_QUESTIONS)}]")
    selected = random.Random(seed).sample(SIMPLE_QUESTIONS, questions)
    return [{"source_id": f"simple-{index:02d}", "question": question} for index, question in enumerate(selected)]
