import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

COMPLETE_TEMPLATES = [
    "Could you help me book a flight to Chicago tomorrow morning?",
    "I think that sounds like a great idea.",
    "No thank you, that will be all for today.",
    "What time does the conference start on Monday?",
    "Yeah, exactly, that's what I was thinking.",
    "Let's reschedule our one-on-one to Thursday afternoon.",
    "I already submitted the pull request for review.",
    "Can you check if the server is responding to health checks?",
    "I'm not really sure that approach will scale well.",
    "That makes complete sense, thanks for explaining.",
    "I'll send you the agenda before the meeting starts.",
    "Please confirm receipt of this email at your earliest convenience.",
    "The quarterly results were better than we expected.",
    "Can we move the deadline to next Friday instead?",
    "I've already updated the documentation with those changes.",
    "Thank you for your time and consideration.",
    "Let me know if you need anything else from me.",
    "The project is on track for a Q3 release.",
    "I don't think that's the right approach for this use case.",
    "Could you please review the attached proposal?",
]

INCOMPLETE_TEMPLATES = [
    "Could you help me book a flight to...",
    "I think that sounds like...",
    "Well, if we decide to go with the first option, then we might...",
    "What I was actually trying to say is that...",
    "Wait, let me pull up the document because...",
    "Can you check if...",
    "Yeah, but on the other hand...",
    "I was thinking maybe around three or...",
    "So if the client calls back before we...",
    "Um, hold on, I'm trying to remember the...",
    "The real issue here is that we haven't...",
    "I wanted to ask you whether...",
    "It depends on whether the team can...",
    "Actually, now that I think about it...",
    "Can we reschedule the call to...",
    "I'm not entirely convinced that...",
    "The problem with that approach is...",
    "Let me just double-check whether...",
    "If the budget allows us to...",
]

FILLER_WORDS = ["uh", "um", "like", "you know", "well"]

def augment_incomplete(text: str) -> str:
    """Truncate a complete sentence at a random point to simulate an incomplete turn."""
    tokens = text.rstrip(".?!").split()
    if len(tokens) > 3:
        cut_point = random.randint(2, len(tokens) - 1)
        truncated = " ".join(tokens[:cut_point])
        if random.random() < 0.3:
            truncated += f" {random.choice(FILLER_WORDS)}"
        return truncated
    return text

def build_dataset(samples_per_class: int = 2500):
    """Generate a stratified train/validation/test split of turn detection samples."""
    dataset = []
    
    # 1. Complete turns (Class 1) from templates
    for _ in range(samples_per_class):
        base = random.choice(COMPLETE_TEMPLATES)
        dataset.append({"text": base, "label": 1})
        
    # 2. Incomplete turns (Class 0) via truncation of complete templates
    for _ in range(samples_per_class):
        base = random.choice(COMPLETE_TEMPLATES)
        incomplete_text = augment_incomplete(base)
        dataset.append({"text": incomplete_text, "label": 0})
        
    # 3. Additional hand-crafted incomplete fragments
    for item in INCOMPLETE_TEMPLATES:
        dataset.append({"text": item.rstrip("."), "label": 0})

    # Train / Val / Test Split (70 / 15 / 15) with stratification
    train_val, test = train_test_split(
        dataset, test_size=0.15, random_state=42,
        stratify=[d['label'] for d in dataset]
    )
    train, val = train_test_split(
        train_val, test_size=0.1765, random_state=42,
        stratify=[d['label'] for d in train_val]
    )

    output_dir = Path("iteration_1/data")
    output_dir.mkdir(exist_ok=True)
    
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        with open(output_dir / f"{split_name}.jsonl", "w") as f:
            for row in split_data:
                f.write(json.dumps(row) + "\n")
                
    print(f"Dataset generated: Train={len(train)}, Val={len(val)}, Test={len(test)}")

if __name__ == "__main__":
    build_dataset()