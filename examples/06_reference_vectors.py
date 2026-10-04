"""Create language-neutral reference test vectors (JSON) to validate other ARAIM implementations."""
from pyaraim.testing.refcases import check_reference_cases, generate_reference_cases

generate_reference_cases("araim_reference_cases.json", n_random=20, seed=2022)
print(check_reference_cases("araim_reference_cases.json"))
