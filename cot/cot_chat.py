# ----------------------------------------------------------------------------------
# - Author Contact: wei.zhang, zwpride@buaa.edu.cn (Original code)
# ----------------------------------------------------------------------------------

from openai import OpenAI
import os

OPENAI_API_KEY =  os.environ.get('OPENAI_API_KEY')

message = [
    {"role": "system", "content": "You are a helpful assistant."}
]

def chat(message):
    client = OpenAI(
        api_key=OPENAI_API_KEY,
    )

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=message,
        temperature=0.1,
    )
    return completion.choices[0].message.content


def output_answer(answer):
    print(answer)

def qa(question):
    message.append({"role": "user", "content": question})
    answer = chat(message)
    message.append({"role": "assistant", "content": answer})
    return answer

def cot(source, debug=False):
    # structure QA
    structure_question = f"Given several log templates and corresponding logs, identify structure including fixed part and <*> part.\n {source}"
    structure_answer = qa(structure_question)
    if debug:
        output_answer(structure_answer)
    # semantics QA
    semantic_question = f"based on log templates with logs and the structure description above, semantic analysis is performed as a whole and each part for each log template above.\n"
    semantic_answer = qa(semantic_question)
    if debug:
        output_answer(semantic_answer)
    # solution QA
    solution_question = f"""based on log templates with logs, the log template structure description and semantics analysis above,  determine whether several given templates can be merged.
Output format:
Merge Solution: yes/no
Merge template: XXX (no new tokens are allowed to be created, keep the num of words)
Reason for Solution: XXX"""
    solution_answer = qa(solution_question)
    output_answer(solution_answer)


if __name__ == "__main__":
    # HDFS
    # templates = ["BLOCK* ask <*> to delete <*>", "BLOCK* ask <*> to delete <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*>", "BLOCK* ask <*> to delete <*> <*> <*> <*> <*> <*> <*> <*> <*>"]
    # logs = [
    #     ["BLOCK* ask 10.250.18.114:50010 to delete  blk_-5140072410813878235", "BLOCK* ask 10.251.122.79:50010 to delete  blk_8048594464172649365"],
    #     ["BLOCK* ask 10.250.17.177:50010 to delete  blk_-8570780307468499817 blk_-9122557405432088649 blk_-4393063808227796056 blk_8767569714374844347 blk_7079754042611867581 blk_7608961006114219538 blk_-5017273584996436939 blk_-6537833125980536955 blk_7610838808763810123 blk_3300803097775546532 blk_-5120750586032922592 blk_1577274266662884430 blk_765879159867598347 blk_-9076085976403711202 blk_-3198963348573340497 blk_-4645750029177277209 blk_-5136142986912961316 blk_5677959846373741243 blk_2107477892986152528 blk_-4235116161537008844 blk_6082535783543982566 blk_-4809870147222033236 blk_8818706925296961012 blk_-5203577173046267127 blk_189089569009261656 blk_446299976487589160 blk_-3916247521166632303 blk_-3324962406687427922 blk_-1807424528783081572 blk_-6858401049333055963 blk_6036564204960295926 blk_-8140723044408248078 blk_-3800132731140204959 blk_1716344083117307767 blk_-5194808114606613364 blk_-5473871016976323232 blk_2920934363167004552 blk_8736689095894369097 blk_-7642734632751940776 blk_3408482260833769309 blk_118013751374560901 blk_7963891081239759520 blk_3813114133944383323 blk_3042818489384932576 blk_-4570173726231458270 blk_-1564644006975920581 blk_338095650783321996 blk_3150135312641203550 blk_4285859645577726288 blk_3438772130782939627 blk_2634772258588877972 blk_-6795664812575964130 blk_3923069610304693233 blk_-1782996202120067721 blk_2004418049430157212 blk_1932147224007687756 blk_-582901062969027153 blk_5072240701440032119 blk_-7919006477393039068 blk_-7318022361288598312 blk_-6974693594143537436 blk_-5435767047126325206 blk_-5805500288959332434 blk_-7109885589081848850 blk_2161580591957523893 blk_7240227881194993860 blk_-8298405680648445349 blk_-4253026248821272215 blk_8377661448601579317 blk_8029153852899017155 blk_-8754388319080705916 blk_-7844092300527332901 blk_710178463364063355 blk_-5136849989188547884 blk_8393887138377503163 blk_-6950176077776664217 blk_-6488701068659548195 blk_2537458728254532453 blk_364441107933628577 blk_6207861897580168557 blk_8814943807366894581 blk_-4150682644311695471 blk_9174833667156726933 blk_649427218152856001 blk_-7403541028238011236 blk_-334982586592048773 blk_61908781908925992 blk_6385574357371832424 blk_-66376131060945541 blk_1372596948297458670 blk_-3389135155401857220 blk_-6035411221441929663 blk_-5127580069634421247 blk_-5685246533892022418 blk_4977937528993040451 blk_5680538862600094527 blk_-8378747462487962732 blk_425101290285860876 blk_6306622708327890839 blk_-1067866602168873257"],
    #     ["BLOCK* ask 10.251.126.5:50010 to delete  blk_-9016567407076718172 blk_-8695715290502978219 blk_-7168328752988473716 blk_-4355192005224403537 blk_-3757501769775889193 blk_-154600013573668394 blk_167132135416677587 blk_2654596473569751784 blk_5202581916713319258"]
    # ]
    # Hadoop
    # templates = ["attempt_ <*> <*> m_ <*> <*> TaskAttempt Transitioned from NEW to UNASSIGNED", "attempt_ <*> <*> r_ <*> <*> TaskAttempt Transitioned from NEW to UNASSIGNED"]
    # logs = [
    #     ["attempt_1445144423722_0020_m_000001_0 TaskAttempt Transitioned from NEW to UNASSIGNED", "attempt_1445144423722_0020_m_000000_0 TaskAttempt Transitioned from NEW to UNASSIGNED", "attempt_1445144423722_0020_m_000003_0 TaskAttempt Transitioned from NEW to UNASSIGNED"],
    #     ["attempt_1445144423722_0020_r_000000_0 TaskAttempt Transitioned from NEW to UNASSIGNED"],
    # ]
    # templates = ["adding path spec: /ws/*", "adding path spec: /mapreduce/*"]
    # logs = [
    #     ["adding path spec: /ws/*"],
    #     ["adding path spec: /mapreduce/*"]
    # ]
    #

    templates = [
        "Server environment: os.verision = Linux <*>",
        "Server environment: os.verision = Windows <*>",
        "Server environment: os.verision = MacOS <*>"
    ]
    logs = [
        [
            "Server environment: os.verision = Linux 5.6.19",
            "Server environment: os.verision = Linux 5.4.263",
            "Server environment: os.verision = Linux 5.9.16"
        ],
        [
            "Server environment: os.verision = Windows 18363.836", 
            "Server environment: os.verision = Windows 16299.1776",
            "Server environment: os.verision = Windows 17134.1304"
        ],
        [
            "Server environment: os.verision = MacOS 10.15.7",
            "Server environment: os.verision = MacOS 11.7.10", 
            "Server environment: os.verision = MacOS 12.2.1"
        ]
    ]

    source = ""
    for i in range(len(templates)):
        source += f"Log template {i+1}: {templates[i]}\n"
        for j in range(len(logs[i])):
            source += f"Log {j+1} corresponding to log template {i+1}: {logs[i][j]}\n"
    print(source)
    cot(source)
