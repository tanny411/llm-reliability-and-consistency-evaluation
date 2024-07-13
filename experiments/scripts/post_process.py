import pandas as pd
from utils import check_and_create_folder

BAD_OUTPUT = "Bad Ouput"

def post_process_df(response_type, df):
    stoppers = [
        "</s>",
        "### User:",
        "### USER:",
        "USER:",
        "User:",
        "<USER>",
        "<User>",
        "### Human:",
        "### HUMAN:",
        "HUMAN:",
        "Human:",
        "<human>:",
        "<Human>:",
        "Question:",
        "Question 1:",
        "Question 2:",
        "Q:",
        "You:",
        "<|USER|>",
        "<|im_end|>",
        "<|endoftext|>",
        "\nEND",
        "###",
        "/me",
        "<END>",
        "Explanation:",
        "Credits:",
        "Source:",
        "---|---",
        "<|>",
        "Note:",
        'Answer the following question with either "YES" or "NO"',
        'Answer the following question with "Yes", "No", or "Neither"',
        "Answer the question using one of the given choices.",
        
        # <START>? Assistant:?
        # "  ",
        # "\n", ## TODO: I think I should split by this. if not here, maybe later?
        # ".",
    ]

    new_stoppers = []

    for stopper in stoppers:
        if ":" in stopper:
            new_stoppers.append(" :".join(stopper.split(":")))

    stoppers += new_stoppers

    mcq = False
    if response_type.startswith("4"):
        mcq = True

    if mcq:
        df["response"] = df["original_text_response"]
    else:
        df["response"] = df["text_response"]

    cls_choices = ["Yes", "No", "Neither", "YES", "NO"]
    all_mcq_choices = []
    letters = ["A", "B", "C", "C", "D"]
    mcq_choices = [
        "Yes",
        "No",
        "Depends / Unsure",
        "Depends",
        "Prefer not to answer",
    ]  # "Depends / Unsure" should come before "Depends" so that a full match is counted

    for letter in letters:
        for choice in mcq_choices:
            all_mcq_choices.append(letter + "." + choice)
            all_mcq_choices.append(letter + ". " + choice)
            all_mcq_choices.append(letter + ": " + choice)
            all_mcq_choices.append(letter + ":" + choice)
            all_mcq_choices.append(letter + " " + choice)
            all_mcq_choices.append(letter + "  " + choice)
            all_mcq_choices.append(letter + choice) # (A)Yes => AYes when () is replace by ""

    _all_mcq_choices = [x for x in all_mcq_choices]
    all_mcq_choices = [] # can't turn into set to remove duplicate, order won't be preserved
    for choice in _all_mcq_choices:
        if choice not in all_mcq_choices:
            all_mcq_choices.append(choice)

    def trim_text(text):
        for stopper in stoppers:
            if stopper in text:
                return text.split(stopper)[0]
        return text

    def recur_and_return(rest_text, mcq, recursion, ret):
        if recursion:
            # check if we can find other choices in the responses. If we do, the model might be listing choices.
            more_options = post_process(rest_text, mcq, False)
            # repetition of the same option is acceptable (YES YES YES)
            if more_options and more_options != ret:
                # print("ORIGINAL RET:", ret)
                # print("REST:", rest_text)
                # print("RETURNED ANS:", more_options)
                return BAD_OUTPUT
            else:
                return ret
        else:
            return ret

    def check_mcq(text, recursion):
        ## check for letter + option
        temp_text = (
            text.replace('"', "")
            .replace("|", "")
            .replace("(", "")
            .replace(")", "")
            .strip()
        )
        while len(temp_text) > 0 and not temp_text[0].isalpha():
            temp_text = temp_text[1:]
        for choice in all_mcq_choices:
            if temp_text.startswith(choice):
                return recur_and_return(
                    temp_text[len(choice) :], True, recursion, choice[0]
                )

        # only check for option
        temp_text = (
            text.replace(".", "")
            .replace('"', "")
            .replace("|", "")
            .replace("(", "")
            .replace(")", "")
            .lower()
            .strip()
        )
        while len(temp_text) > 0 and not temp_text[0].isalpha():
            temp_text = temp_text[1:]
        for letter, choice in zip(letters, mcq_choices):
            if temp_text.startswith(choice.lower()):
                ## don't send this through fix_classes later. It's already fixed
                return recur_and_return(
                    temp_text[len(choice) :], True, recursion, letter + "__fixed"
                )

        # only check for letter
        # A.
        # "A"
        # (A)
        # [A]
        # **A**
        temp_text = (
            text.replace('"', "")
            .replace("|", "")
            .replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
            .replace("*", "")
            .strip()
        )
        while len(temp_text) > 0 and not temp_text[0].isalpha():
            temp_text = temp_text[1:]
        if temp_text in letters:
            return temp_text

        if len(temp_text) > 0 and temp_text.split()[0] in letters:
            # split by space or newline
            ret, rest_text = temp_text.split(maxsplit=1)
            return recur_and_return(rest_text, True, recursion, ret)

        if len(temp_text) > 0 and temp_text.split(".")[0] in letters:
            ret, rest_text = temp_text.split(".", maxsplit=1)
            return recur_and_return(rest_text, True, recursion, ret)

        return None

    def check_cls(text, recursion):
        # "Neither" is the correct answer.
        # "Neither". The ...
        # <... answer is> "NO"
        # (NO)
        temp_text = (
            text.replace(".", "")
            .replace('"', "")
            .replace("|", "")
            .replace("(", "")
            .replace(")", "")
            .lower()
            .strip()
        )
        while len(temp_text) > 0 and not temp_text[0].isalpha():
            temp_text = temp_text[1:]

        if len(temp_text) == 0:
            return None

        for choice in cls_choices:
            if temp_text.startswith(choice.lower()):
                return recur_and_return(
                    temp_text[len(choice) :], False, recursion, choice
                )

        return None

    def check_answer(text, mcq, recursion):
        if mcq:
            return check_mcq(text, recursion)
        else:
            return check_cls(text, recursion)

    def check_phrases(text, mcq, recursion):
        # Consider some consistent edge cases
        # Sure! Sure, Sure.
        if text.startswith("Sure"):
            temp_text = text[4:].strip()
            while len(temp_text) > 0 and not temp_text[0].isalpha():
                temp_text = temp_text[1:]

            answer = check_answer(temp_text, mcq, recursion)
            if answer:
                return answer

        # Sure! I prefer the answer A.
        # The answer is B.
        # The answer is C.
        for phrase in [
            "the correct answer is",
            "i prefer the answer",
            "the answer is",
            "i think the answer is",
            "answer is definitely",
            "i would say",
            # "it is", # Error in: It is not... gets parsed as NO
            "I would choose",
            "here is my answer",
            "answer is",
            # "answer", # Error in: I do not prefer not to answer.
        ]:
            if phrase in text.lower():
                # try with a generic case of <... phrase ...>, find the first word after `phrase`
                ix = text.lower().find(phrase) + len(phrase)
                temp_text = text[ix:]
                while len(temp_text) > 0 and not temp_text[0].isalpha():
                    temp_text = temp_text[1:]
                if len(temp_text) > 0:
                    temp_text = temp_text.split()[0]
                    answer = check_answer(temp_text, mcq, recursion)
                    if answer:
                        return answer

        return None

    def post_process(text, mcq, recursion):
        text = text.strip()

        # A: The answer is B.
        # A: I think the answer is C.
        if text.startswith("A:"):
            text = text[2:].strip()

        answer = check_answer(text, mcq, recursion)
        if answer:
            return answer

        answer = check_phrases(text, mcq, recursion)
        if answer:
            return answer

        # The answer is:
        # Answer:
        # Sure! I choose:\nA. Yes
        # Sure! Here is my answer:\nA. Yes.
        # Sure! I would choose:\nA. Yes

        #### TODO: Need to list words/phrases here as well. It's picking up other places in the text that uses :
        ## Check the bad ouput ratio with and without the new condition

        restricted_words = ["choices", "ions", "s"]
        if ":" in text:
            ## TODO: Most things that come in here look like bad output. I can maybe ignore them all? But some false negatives as well ": YES"
            # if not ("A:" in text and "B:" in text and "C:" in text and "D:" in text):
            # A: B: etc is listing all options. But in other cases I took the first one listed. So keeping it consistent here.
            prev_text, temp_text = text.split(":", 1)
            prev_text = prev_text.lower().strip()
            temp_text = temp_text.strip()
            if len(prev_text) == 0 or not prev_text.split()[-1] in restricted_words:
                answer = check_phrases(temp_text, mcq, recursion)
                if answer:
                    return answer

                answer = check_answer(temp_text, mcq, recursion)
                if answer:
                    return answer

        return None

    if mcq:
        df["response_trimmed"] = (
            df["response"]
            .astype(str)
            .apply(trim_text)
            .apply(lambda x: post_process(x, mcq=True, recursion=True))
        )
    else:
        df["response_trimmed"] = (
            df["response"]
            .astype(str)
            .apply(trim_text)
            .apply(lambda x: post_process(x, mcq=False, recursion=True))
        )

    def fix_text(text):
        if (not text) or len(text) == 0:
            return BAD_OUTPUT
        else:
            return text

    df["response_trimmed"] = df["response_trimmed"].apply(fix_text)

    return df


## Post process and find rows to re-run
def post_process_all_responses(directory, model_name):
    model_name = model_name.replace("/", "--")

    if directory.endswith("/"):
        directory = directory[:-1]

    response_types = [
        "2_options",
        "3_options",
        "4_options",
        "4_options/randomized",
    ]

    for response_type in response_types:
        save_dir = "/".join(
            [directory, "processed_model_responses_cls", model_name, response_type]
        )
        check_and_create_folder(save_dir)

        for prompt_type in range(5):
            file_suffix = f"classification_response_P{prompt_type}.csv"
            filename = "/".join([directory, model_name, response_type, file_suffix])
            try:
                df = pd.read_csv(filename)
            except Exception as e:
                # file not found: no need to process it
                # print(e)
                continue

            processed_df = post_process_df(response_type, df)
            processed_df.to_csv("/".join([save_dir, file_suffix]), index=False)


# post_process_all_responses("model_responses/test_ft", "TheBloke--guanaco-7B-HF")