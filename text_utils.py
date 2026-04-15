import re


FORBIDDEN_TAGS = [
    "thought", "references", "conclusion", "answer",
    "response", "output", "result", "context", "question",
]


def clean_response(text: str) -> str:
    """불필요한 XML 태그를 제거하고 과도한 빈 줄을 정리합니다."""
    for tag in FORBIDDEN_TAGS:
        text = re.sub(rf"</?{tag}>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_think_and_answer(text: str):
    """문자열에서 <think> 부분과 실제 답변 부분을 완벽하게 분리합니다."""
    if "<think>" in text and "</think>" in text:
        parts = text.split("</think>", 1)
        think_content = parts[0].split("<think>")[-1].strip()
        answer_content = clean_response(parts[1])
        return think_content, answer_content
    elif "<think>" in text:
        think_content = text.split("<think>")[-1].strip()
        return think_content, ""
    else:
        return "", clean_response(text)
