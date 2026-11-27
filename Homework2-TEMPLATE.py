# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
# torch: the deep-learning framework the model runs on.
# transformers: Hugging Face library that gives us the model + tokenizer.
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, rdMolDescriptors

# ---------------------------------------------------------------------------
# Model + device configuration
# ---------------------------------------------------------------------------
# The Hugging Face identifier for the model we're going to download and run.
model_id = "Qwen/Qwen3-4B"

# Pick the best available hardware to run the model on:
#   - "mps"  = Apple Silicon GPU (M1/M2/M3 Macs)
#   - "cuda" = NVIDIA GPU
#   - "cpu"  = fallback if neither GPU type is available
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# System prompt  <-- EDIT ME to change the model's persona / behavior
# ---------------------------------------------------------------------------
# The system prompt is sent to the model before every user question. Use it
# to set tone, role, constraints, output format, etc.

# TODO 1: Write a system prompt here giving instructions to the
# AI model on how to handle the user's input.
SYSTEM_PROMPT = '''
You are a routing layer for a chemistry Q&A agent. You do NOT answer chemistry
questions yourself — your only job is to read the user's message and output a
JSON object describing what they want, so that downstream code can call RDKit
to compute the real answer.

The agent supports exactly these 8 properties, each computed from a SMILES string:
- molecular_weight
- logp              (lipophilicity / hydrophobicity)
- tpsa               (topological polar surface area)
- h_bond_donors
- h_bond_acceptors
- rotatable_bonds
- num_rings
- num_aromatic_rings

Read the user's message and extract every (SMILES string, property) pair being
asked about. Users may:
- misspell words ("lphilicity", "molecular wght")
- use unusual punctuation, spacing, or capitalization
- phrase it as a question, a command, or a sentence fragment
- ask about multiple properties and/or multiple SMILES strings in one message
Interpret their intent rather than matching exact wording.

Respond with ONLY a single JSON object, no other text, in exactly this shape:

{
  "queries": [
    {"smiles": "<the SMILES string as written>", "property": "<one of the 8 keys above>"}
  ],
  "off_topic": <true if the message asks about something other than these 8
                properties and is not just missing a SMILES>,
  "no_smiles": <true if the user clearly wants one of the 8 properties but gave
                no SMILES string at all>
}

Rules:
- Do not validate or guess whether a SMILES string is chemically valid — just
  extract it as written. Validation happens downstream.
- Do not compute or state any numeric answer yourself.
- If a message asks for a property not in the list (e.g. "what's its boiling
  point?"), do not include it in "queries" — set "off_topic": true instead.
- If nothing in the message relates to the 8 properties, "queries" should be
  an empty list and "off_topic": true.
- Never include explanations, markdown, or text outside the JSON object.
'''

# ---------------------------------------------------------------------------
# Load the tokenizer and model
# ---------------------------------------------------------------------------
# Tokenizer: converts text <-> token IDs the model understands.
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Model: downloads the weights (first run only) and moves them onto `device`.
# Pick a precision that works on the current hardware:
#   - MPS (Apple Silicon): float16 (bfloat16 is not supported on MPS)
#   - CUDA (NVIDIA GPU):   float16 (works on all CUDA GPUs; saves memory)
#   - CPU (Windows/Linux/Mac without a GPU): float32 (half precision is
#     often slow or unsupported for many ops on CPU)
if device == "cpu":
    dtype = torch.float32
else:
    dtype = torch.float16
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)

# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------
# Keep asking the user for prompts until they type "quit" or "exit"
# (or press Ctrl+C). Each turn is independent — the model does NOT remember
# previous questions in this loop, since we rebuild `messages` from scratch
# every iteration.
while True:
    prompt = input("\nPrompt (type 'quit' to exit): ")
    if prompt.strip().lower() in {"quit", "exit"}:
        break

    # -----------------------------------------------------------------------
    # Prepare the prompt for the model
    # -----------------------------------------------------------------------
    # 1) Build the chat as a list of role/content dicts. The system message
    #    comes first so the model reads it before the user's question.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # 2) Wrap the messages in the model's chat template (adds the special
    #    role/turn tokens Qwen expects). `add_generation_prompt=True` appends
    #    the marker that tells the model "your turn to reply."
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    # 3) Turn the formatted text into tensors of token IDs, and move them
    #    onto the same device as the model.
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # -----------------------------------------------------------------------
    # Generate a response
    # -----------------------------------------------------------------------
    # `inference_mode` disables gradient tracking -> faster and uses less
    # memory since we're only doing a forward pass, not training.
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=256)

    # -----------------------------------------------------------------------
    # Decode and print just the newly generated text
    # -----------------------------------------------------------------------
    # `output` contains the prompt tokens followed by the generated tokens.
    # Slicing from `inputs.input_ids.shape[1]:` skips the prompt so we only
    # print what the model actually wrote. `skip_special_tokens=True` strips
    # chat-template markers from the printed string.
    response = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # TODO 1.5: Make sure you delete this line of code before submitting.
    # This line of code is simply here to help you to debug.
    # print('LLM response so far: ', response) 

    # -----------------------------------------------------------------------
    # Do your work to do the tool calls based on LLM response
    # -----------------------------------------------------------------------
    
    # TODO 2: Based on the output of the LLM, make the appropriate tool
    # calls to RDKit and output the response in a user friendly manner.
    # Have fun in this part of the code!

  #Parse the LLM response and route requests to RDKit
    # map each property name used by Qwen to:
    #   1) a user-friendly display name
    #   2) the appropriate RDKit calculation
    property_tools = {
        "molecular_weight": ("Molecular weight", lambda mol: Descriptors.MolWt(mol)),
        "logp": ("LogP", lambda mol: Crippen.MolLogP(mol)),
        "tpsa": ("TPSA", lambda mol: rdMolDescriptors.CalcTPSA(mol)),
        "h_bond_donors": ("Hydrogen bond donors", lambda mol: Lipinski.NumHDonors(mol)),
        "h_bond_acceptors": ("Hydrogen bond acceptors", lambda mol: Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": ("Rotatable bonds", lambda mol: Lipinski.NumRotatableBonds(mol)),
        "num_rings": ("Number of rings", lambda mol: Lipinski.RingCount(mol)),
        "num_aromatic_rings": ("Number of aromatic rings", lambda mol: Lipinski.NumAromaticRings(mol)),
    }

    try:
        parsed_response = json.loads(response)

    except json.JSONDecodeError:
        print(
            "Sorry, I couldn't understand that request. "
            "Please ask about a supported chemical property and provide a SMILES string."
        )
        continue

    # requests with no smiles string
    if parsed_response.get("no_smiles", False):
        print("Please provide a valid SMILES string.")
        continue

    queries = parsed_response.get("queries", [])

    # requests that are out of scope
    if not queries and parsed_response.get("off_topic", False):
        print(
            "I can only answer questions about molecular weight, logP, TPSA, "
            "hydrogen bond donors, hydrogen bond acceptors, rotatable bonds, "
            "number of rings, and number of aromatic rings."
        )
        continue

    # smiles : requested property
    for query in queries:

        smiles = query.get("smiles")
        property_name = query.get("property")

        if not smiles:
            print("Please provide a valid SMILES string.")
            continue

        if property_name not in property_tools:
            print(
                f"I don't support the property '{property_name}'. "
                "Please ask about one of the 8 supported properties."
            )
            continue

        # convert smiles into rdkit molecule
        # Chem.MolFromSmiles returns None if rdkit cannot parse the SMILES
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            print(
                f"'{smiles}' is not a valid SMILES string. "
                "Please provide a valid SMILES string."
            )
            continue

        # calling appropriate rdkit tool
        display_name, tool_function = property_tools[property_name]
        result = tool_function(mol)

        if property_name == "molecular_weight":
            print(
                f"{display_name} of {smiles}: "
                f"{result:.2f} g/mol"
            )

        elif property_name == "logp":
            print(
                f"{display_name} of {smiles}: "
                f"{result:.2f}"
            )

        elif property_name == "tpsa":
            print(
                f"{display_name} of {smiles}: "
                f"{result:.2f} Å²"
            )

        else:
            # These properties return counts.
            print(
                f"{display_name} of {smiles}: "
                f"{int(result)}"
            )

    # if on topic and valid query but has unsupported version, let user know what part is unsupported
    if parsed_response.get("off_topic", False) and queries:
        print(
            "Note: Part of your request was outside the properties I support. "
            "I can answer molecular weight, logP, TPSA, hydrogen bond donors, "
            "hydrogen bond acceptors, rotatable bonds, number of rings, and "
            "number of aromatic rings."
        )

