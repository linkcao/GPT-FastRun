import os
import json
import time
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
import openai
import json
from usage import read_config
import threading

logger = logging.getLogger(__name__)
fail_key = []
result_global = []

def read_prompts(path):
    sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            sentences.append(line.strip())
    return sentences

def gpt_run(prompts, api_key, cfg):
    # 20个一组
    result = []
    for prompt in prompts:
        retry_count = 0
        retry_interval = 2  # 重试间隔时间
        while retry_count < 5:
            try:
                response = openai.Completion.create(model=cfg['engine'], prompt=prompt.strip(), temperature=0, max_tokens=7, api_key = api_key)
                item = {'prompt': prompt, 'result': response['choices'][0]['text']}
                print('current thread:' + threading.currentThread().getName())
                print('prompt:' + prompt)
                print('response:' + response['choices'][0]['text'].strip())
                print('\n')
                result.append(item) 
                time.sleep(10)
                break
            except openai.error.RateLimitError as e:
                print("超出openai api 调用频率：", e)
                retry_count += 1
                retry_interval *= 2  # 指数退避策略，每次重试后加倍重试间隔时间
                time.sleep(retry_interval)
            except Exception as e:
                print("任务执行出错：", e)
                retry_count += 1
                retry_interval *= 2  # 指数退避策略，每次重试后加倍重试间隔时间
                time.sleep(retry_interval)
        if retry_count > 5:
            fail_key.append(api_key)
            return
    with open('result.txt', 'a') as f:
        for item in result:
            f.write(json.dumps(item))
            # 当前的线程名称
            f.write('\n')
    return result

def get_result(future):
    result_global.extend(future.result())

def main():
    cfg = read_config()
    api_keys= cfg['checked_keys']
    prompts = read_prompts(cfg['prompt_path'])
    os.environ["https_proxy"] = cfg['proxy']

    # split the prompts into batch
    batch = min(cfg['batch'], len(api_keys))
    prompts_batch = [prompts[i:i+int(len(prompts)/batch)] for i in range(0, len(prompts), int(len(prompts)/batch))]

    # 创建一个包含2条线程的线程池
    with ThreadPoolExecutor(max_workers=batch,thread_name_prefix="gpt_") as threadPool:
        for index, prompts in enumerate(prompts_batch):
            # free trial users rate limits 20 requests / minute
            future = threadPool.submit(gpt_run, prompts, api_keys[index], cfg)
            future.add_done_callback(get_result)
    while( len(result_global) < len(prompts)) :
        miss_prompt = []
        response_prompt = []
        for result in result_global:
            response_prompt.append(result['prompt'])
        for prompt in prompts:
            if prompt not in response_prompt:
                miss_prompt.append(prompt)
        for key in api_keys:
            if key not in fail_key:
                miss_result = gpt_run(key, miss_prompt)
                result_global.extend(miss_result)
    return result_global


if __name__ == '__main__':
    main()
