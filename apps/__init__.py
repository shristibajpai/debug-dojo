
from apps.llm_service import ask_qwen

answer = ask_qwen(
    "In a loop using range(len(arr) + 1), "
    "what boundary issue should I investigate? "
    "Give one hint without corrected code."
)

print("\nDebug Dojo response:\n")
print(answer)
