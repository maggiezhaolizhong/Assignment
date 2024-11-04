import sys
import os
import openai
import datasets
import asyncio
import argparse
from tqdm.asyncio import tqdm_asyncio


def get_prompt_template():
    print("Enter a prompt:")
    user_input = sys.stdin.readline().strip()
    if not user_input:
        user_input = """Answer the question: """
    #  print(user_input)
    return user_input

# get response from AI model
async def get_llm_response_async(prompt, prompt_template, model):
    formatted_prompt = prompt_template + prompt
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    # print(f"openai_api_key:{openai_api_key}")
    client = openai.AsyncOpenAI(api_key=openai_api_key,
                                base_url="https://api.moonshot.cn/v1"
                                )

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": formatted_prompt}
        ],
        max_tokens=500,  # Adjust as needed
        n=1,
        stop=None,
        temperature=0.0,  # 0-1
    )

    return response.choices[0].message.content.strip()

def evaluate(response, target):
    response_label = response.split()[0].strip("() ")  # Extract the answer label
    target_label = target.strip("() ")
    return response_label == target_label

async def main(args):
    dataset = datasets.load_dataset("lukaemon/bbh", "reasoning_about_colored_objects")
    subset = dataset["test"].shuffle(seed=args.seed).select(range(args.limit))

    # for item in subset:
    #   print(item)
    # example : {"input":xxxx,"target":(F)}

    prompt_template = get_prompt_template()  # user_input or "Answer the question:"

    semaphore = asyncio.Semaphore(args.concurrency)

    from pprint import pprint
    async def process_example(example):
        async with semaphore:
            input_ = example["input"]
            target = example["target"]
            response = await get_llm_response_async(input_, prompt_template, args.model)
            is_correct = evaluate(response, target)
            result = {
                'question': input_,
                'ground_truth': target,
                'model_response': response,
                'is_correct': is_correct
            }
            # if not is_correct:
            #   print(result)
            return result

    tasks = [process_example(example) for example in subset]  # result list

    evaluation_results = await tqdm_asyncio.gather(*tasks, desc="Processing examples")

    # Calculate and print the metrics
    accuracy = 0
    # calculate accuracy
    example_pass = sum(1 for res in evaluation_results if res["is_correct"])
    accuracy = example_pass / len(tasks) * 100
    print(f"Accuracy: {accuracy:.2f}")

    # implement any other metric that you think might be usefull


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--limit", type=int, default=100)  # limit 100 examples
    # args.add_argument("--model", type=str, default="gpt-3.5-turbo")
    # models "moonshot-v1-8k","moonshot-v1-32k","moonshot-v1-128k"
    args.add_argument("--model", type=str, default="moonshot-v1-8k")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--concurrency", type=int, default=5)
    args = args.parse_args()

    asyncio.run(main(args))
