#!/usr/bin/env python3
"""
Inference script for HuggingFace model on Ballerina problems using Unsloth.
Generates 5 completions per problem and evaluates them using test cases.
Optimized for 16GB Nvidia P100 GPU.
"""

import json
import base64
import zlib
import pickle
import subprocess
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch

# Configuration
INPUT_FILE = Path("ballerina_grpo_X4.json")
OUTPUT_FILE = Path("ballerina_grpo_X4_inference_results.json")
MODEL_NAME = "didula-wso2/exp_23_emb_grpo_checkpoint_1000_16bit_vllm"
NUM_GENERATIONS = 5
THRESHOLD = 0.75  # 75% of test cases must pass

print(f"Input file: {INPUT_FILE}")
print(f"Output file: {OUTPUT_FILE}")
print(f"Model: {MODEL_NAME}")
print(f"Generations per problem: {NUM_GENERATIONS}")
print(f"Test pass threshold: {THRESHOLD*100}%")

# Load model and tokenizer using Unsloth (optimized for 16GB P100 GPU)
print("\nLoading model and tokenizer with Unsloth...")
print("Optimizing for 16GB Nvidia P100 GPU...")
print("Note: Using transformers backend (vLLM disabled to avoid dependency issues)")

max_seq_length = 2048
dtype = None  # Auto-detect best dtype
load_in_4bit = True  # Use 4-bit quantization for 16GB GPU
lora_rank = 16

# Disable vLLM by setting fast_inference=False - this uses standard transformers backend
# which is more compatible and still optimized by Unsloth
# Also set environment variable to prevent vLLM loading
import os
os.environ["UNSLOTH_USE_VLLM"] = "0"  # Explicitly disable vLLM

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    fast_inference=False,  # Disable vLLM, use transformers backend
    max_lora_rank=lora_rank,
)

# Set up chat template (Qwen format)
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
)

# Enable fast inference optimizations (Unsloth optimizations without vLLM)
FastLanguageModel.for_inference(model)  # Enable inference optimizations

if torch.cuda.is_available():
    print(f"Using CUDA (GPU: {torch.cuda.get_device_name(0)})")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("Using CPU")

print("Model loaded successfully with Unsloth optimizations!")

# System prompt for Ballerina code generation (matching generate_corrections.ipynb)
SYSTEM_PROMPT = """You are a pragmatic Ballerina programmer who enjoys test driven development. Given the following question, write a Ballerina script to complete the task and then write the the unit tests to validate the functionality. Also implement a main function check with a user input.

1. Make the code simple and easy to understand.
2. Try to limit library usage to the standard library. Be careful with your types, and try to limit yourself to the basic built in types and standard library functions.
3. Before you start writing the function you can think through how to solve the problem and perform reasoning in the comments above the function.
4. Implement a main function to accept user inputs. Output the outputs of the function.
5. Don't define inputs yourself. Do not hardcode inputs. Only use user inputs.
6. Then write unit tests for the function you defined. Make sure to write at least 4 assertions to test the function. The tests should be a simple.

[Important] Strictly follow the following output format for each response: Make sure to include code inside <CODE> and <TESTS> blocks. Only use Ballerina Programming Language. Never use any other programming languages. Implement proper error handling.

# Overview
Brief overview about the solution.

<CODE>
```ballerina
// Reasoning goes here
// and can be multi-line

import ballerina/io;

function add(int a, int b) returns int {
    return a + b;
}

public function main() {
    string? input_line = io:readln();
    if input_line is string {
        string[] parts = input_line.split(" ");
        if parts.length() == 2 {
            int|error first_num = 'int:fromString(parts[0]);
            int|error second_num = 'int:fromString(parts[1]);

            if first_num is int && second_num is int {
                int result = add(first_num, second_num);
                io:println(result.toString());
            }
        }
    }
}
```
</CODE>

<TESTS>
```ballerina
import ballerina/test;

@test:Config { }
function testAssertEquals() {
    int addResult = add(40, 2);
    test:assertEquals(addResult, 42);

    addResult = add(0, 0);
    test:assertEquals(addResult, 0);

    addResult = add(-1, 1);
    test:assertEquals(addResult, 0);

    addResult = add(-5, -5);
    test:assertEquals(addResult, -10);
}
```
</TESTS>
"""


def decode_answer_field(answer_str: str) -> List[Dict[str, str]]:
    """Decode test cases from answer field: base64 → zlib → pickle → JSON"""
    try:
        # Base64 decode
        decoded_bytes = base64.b64decode(answer_str.encode("utf-8"))
        # Zlib decompress
        decompressed = zlib.decompress(decoded_bytes)
        # Pickle loads
        pickled_data = pickle.loads(decompressed)
        # JSON loads
        test_cases = json.loads(pickled_data)
        return test_cases
    except Exception as e:
        print(f"    Error decoding answer field: {e}")
        return []


def extract_code_from_completion(completion: str) -> Optional[str]:
    """Extract code block from completion following generate_corrections.ipynb pattern"""
    if '<CODE>' in completion and '</CODE>' in completion:
        code_section = completion.split('<CODE>')[1].split('</CODE>')[0]
        # Extract from ballerina code block
        if '```ballerina' in code_section:
            code = code_section.split('```ballerina')[1].split('```')[0].strip()
        elif '```' in code_section:
            code = code_section.split('```')[1].split('```')[0].strip()
        else:
            code = code_section.strip()
        return code
    return None


def normalize_output(output: str) -> str:
    """Normalize output by stripping trailing whitespace from each line and the entire output"""
    lines = output.split('\n')
    normalized_lines = [line.rstrip() for line in lines]
    return '\n'.join(normalized_lines).strip()


def run_test_cases(code: str, test_cases: List[Dict[str, str]], threshold: float = 0.75) -> Tuple[bool, int, int, str, Dict]:
    """Run test cases and check if at least threshold% pass
    Returns: (is_valid, passed, total, validation_msg, error_details)
    error_details includes: compilation_error, failing_test_case
    """
    if not code:
        return False, 0, 0, "No code extracted", {}
    
    # Check for basic Ballerina structure
    if 'function main()' not in code and 'public function main()' not in code:
        return False, 0, 0, "Missing main function", {}
    
    error_details = {}
    
    if not test_cases or len(test_cases) == 0:
        # No test cases, fall back to compilation check
        with tempfile.TemporaryDirectory() as tmpdir:
            code_file = Path(tmpdir) / "main.bal"
            code_file.write_text(code)
            
            try:
                result = subprocess.run(
                    ["bal", "build", "main.bal"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    return True, 0, 0, "Code compiles (no test cases)", {}
                else:
                    error_details['compilation_error'] = result.stderr
                    return False, 0, 0, f"Compilation error: {result.stderr[:500]}", error_details
            except Exception as e:
                error_details['compilation_error'] = str(e)
                return False, 0, 0, f"Validation error: {str(e)}", error_details
    
    # First, try to compile the code
    with tempfile.TemporaryDirectory() as tmpdir:
        code_file = Path(tmpdir) / "main.bal"
        code_file.write_text(code)
        
        # Check compilation first
        compile_result = subprocess.run(
            ["bal", "build", "main.bal"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if compile_result.returncode != 0:
            error_details['compilation_error'] = compile_result.stderr
            return False, 0, len(test_cases), f"Compilation error: {compile_result.stderr[:500]}", error_details
        
        # Code compiles, now run test cases
        passed = 0
        failed = 0
        first_failing_test = None
        
        for idx, test_case in enumerate(test_cases):
            test_input = test_case.get('input', '')
            expected_output = test_case.get('output', '')
            
            # Normalize expected output
            expected_output_normalized = normalize_output(expected_output)
            
            try:
                # Write input to temp file
                input_file = Path(tmpdir) / f"input_{idx}.txt"
                input_file.write_text(test_input)
                
                # Run the code
                with open(input_file, 'r') as inp:
                    result = subprocess.run(
                        ["bal", "run", "main.bal"],
                        cwd=tmpdir,
                        stdin=inp,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                
                if result.returncode != 0:
                    failed += 1
                    # Capture first failing test with runtime error
                    if first_failing_test is None:
                        first_failing_test = {
                            'input': test_input,
                            'expected_output': expected_output,
                            'actual_output': result.stderr if result.stderr else result.stdout,
                            'error': f"Runtime error (exit code {result.returncode})"
                        }
                    continue
                
                # Normalize actual output
                actual_output_normalized = normalize_output(result.stdout)
                
                if actual_output_normalized == expected_output_normalized:
                    passed += 1
                else:
                    failed += 1
                    # Capture first failing test with output mismatch
                    if first_failing_test is None:
                        first_failing_test = {
                            'input': test_input,
                            'expected_output': expected_output,
                            'actual_output': result.stdout,
                            'error': 'Output mismatch'
                        }
                    
            except subprocess.TimeoutExpired:
                failed += 1
                if first_failing_test is None:
                    first_failing_test = {
                        'input': test_input,
                        'expected_output': expected_output,
                        'actual_output': '',
                        'error': 'Timeout (code took >10 seconds)'
                    }
            except Exception as e:
                failed += 1
                if first_failing_test is None:
                    first_failing_test = {
                        'input': test_input,
                        'expected_output': expected_output,
                        'actual_output': '',
                        'error': str(e)
                    }
        
        total = len(test_cases)
        pass_rate = passed / total if total > 0 else 0
        
        if pass_rate >= threshold:
            return True, passed, total, f"Passed {passed}/{total} tests ({pass_rate*100:.1f}%)", {}
        else:
            if first_failing_test:
                error_details['failing_test_case'] = first_failing_test
            return False, passed, total, f"Only passed {passed}/{total} tests ({pass_rate*100:.1f}%), need {threshold*100:.0f}%", error_details


def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int = 2048) -> str:
    """Generate a single completion using Unsloth-optimized model"""
    # Format prompt with system prompt
    full_prompt = f"""{SYSTEM_PROMPT}

Problem:
{prompt}

Please generate Ballerina code that solves the problem above. Make sure to:
1. Read input from stdin (not command line arguments) using io:readln() or similar
2. Process the input according to the problem specification
3. Output the result to stdout using io:println()
4. Handle all edge cases properly
5. Use proper error handling
"""
    
    # Format messages for chat template
    messages = [
        {"role": "user", "content": full_prompt},
    ]
    
    # Use Unsloth's optimized generation
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    # Move inputs to model's device (Unsloth models are already on the correct device)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Use Unsloth's optimized generation with torch.inference_mode for better performance
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,  # Enable KV cache for faster inference
        )
    
    # Decode only the new tokens
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )
    
    return generated_text


def process_problem(problem: Dict[str, Any], model, tokenizer, problem_idx: int) -> Dict[str, Any]:
    """Process a single problem: generate 5 completions and evaluate each"""
    prompt = problem.get('prompt', '')
    answer = problem.get('answer', '')
    
    # Decode test cases from answer field
    test_cases = decode_answer_field(answer)
    if test_cases:
        print(f"  📋 {len(test_cases)} test cases decoded")
    else:
        print(f"  ⚠ No test cases found in answer field")
    
    generations = []
    
    for gen_idx in range(NUM_GENERATIONS):
        print(f"  Generating completion {gen_idx + 1}/{NUM_GENERATIONS}...")
        
        try:
            # Generate completion
            completion = generate_completion(model, tokenizer, prompt)
            
            # Extract code
            code = extract_code_from_completion(completion)
            
            if not code:
                print(f"    ⚠ Could not extract code from completion")
                generations.append({
                    'completion': completion,
                    'code': None,
                    'passed': False,
                    'tests_passed': 0,
                    'tests_total': len(test_cases),
                    'pass_rate': 0.0,
                    'validation_msg': 'Could not extract code'
                })
                continue
            
            # Evaluate test cases
            is_valid, passed, total, validation_msg, error_details = run_test_cases(
                code, test_cases, threshold=THRESHOLD
            )
            
            pass_rate = passed / total if total > 0 else 0.0
            
            generations.append({
                'completion': completion,
                'code': code,
                'passed': is_valid,
                'tests_passed': passed,
                'tests_total': total,
                'pass_rate': pass_rate,
                'validation_msg': validation_msg,
                'error_details': error_details if error_details else None
            })
            
            status = "✓" if is_valid else "✗"
            print(f"    {status} {validation_msg}")
            
        except Exception as e:
            print(f"    ✗ Error generating/evaluating: {e}")
            generations.append({
                'completion': None,
                'code': None,
                'passed': False,
                'tests_passed': 0,
                'tests_total': len(test_cases),
                'pass_rate': 0.0,
                'validation_msg': f'Error: {str(e)}',
                'error_details': None
            })
    
    # Count passing generations
    passing_generations = sum(1 for gen in generations if gen['passed'])
    
    # Create result entry
    result = {
        **{k: v for k, v in problem.items() if k not in ['generations', 'passing_generations']},
        'generations': generations,
        'passing_generations': passing_generations
    }
    
    return result


# Main execution
if __name__ == "__main__":
    # Load problems
    print(f"\nLoading problems from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        problems = json.load(f)
    
    print(f"Loaded {len(problems)} problems")
    print(f"Processing problems 1-{min(5018, len(problems))}...")
    
    # Process problems
    results = []
    stats = {
        'total': 0,
        'passing_1': 0,
        'passing_2': 0,
        'passing_3': 0,
        'passing_4': 0,
        'passing_5': 0,
        'passing_0': 0
    }
    
    for idx, problem in enumerate(problems[:5018]):
        problem_num = idx + 1
        print(f"\n[{problem_num}/5018] Processing problem {idx} (rating: {problem.get('rating', 'unknown')})...")
        
        result = process_problem(problem, model, tokenizer, idx)
        results.append(result)
        
        # Update stats
        passing = result['passing_generations']
        stats['total'] += 1
        if passing == 0:
            stats['passing_0'] += 1
        elif passing == 1:
            stats['passing_1'] += 1
        elif passing == 2:
            stats['passing_2'] += 1
        elif passing == 3:
            stats['passing_3'] += 1
        elif passing == 4:
            stats['passing_4'] += 1
        elif passing == 5:
            stats['passing_5'] += 1
        
        print(f"  Result: {passing}/{NUM_GENERATIONS} generations passed")
        
        # Print progress every 10 problems
        if (idx + 1) % 10 == 0:
            print("\n" + "-" * 80)
            print(f"Progress: {idx + 1}/5018")
            print(f"Stats: 0 passing: {stats['passing_0']}, 1: {stats['passing_1']}, "
                  f"2: {stats['passing_2']}, 3: {stats['passing_3']}, "
                  f"4: {stats['passing_4']}, 5: {stats['passing_5']}")
            print("-" * 80)
    
    # Save results
    print(f"\nSaving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Final summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total problems processed: {stats['total']}")
    print(f"Problems with 0 passing: {stats['passing_0']}")
    print(f"Problems with 1 passing: {stats['passing_1']}")
    print(f"Problems with 2 passing: {stats['passing_2']}")
    print(f"Problems with 3 passing: {stats['passing_3']}")
    print(f"Problems with 4 passing: {stats['passing_4']}")
    print(f"Problems with 5 passing: {stats['passing_5']}")
    print("=" * 80)


