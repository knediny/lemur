<div align= "center">
    <div align= "center">
    <h1><img src="./assets/Lemur-emo.png" width="6%"><span style="font-weight: bold; font-size: larger;">LEMUR</span></h1>
</div>

## Abstract
Logs produced by extensive software systems are integral to monitoring system behaviors. Advanced log analysis facilitates the detection, alerting, and diagnosis of system faults. Log parsing, which entails transforming raw log messages into structured templates, constitutes a critical phase in the automation of log analytics. Existing log parsers fail to identify the correct templates due to reliance on human-made rules. Besides, These methods focus on statistical features while ignoring semantic information in log messages. 
To address these challenges, we introduce a cutting-edge **L**og parsing framework with **E**ntropy sampling and Chain-of-Thought **M**erging (<img src="./assets/Lemur-emo.png" width="3%">**LEMUR**). Specifically, to discard the tedious manual rules. We propose a novel sampling method inspired by information entropy, which efficiently clusters typical logs. Furthermore, to enhance the merging of log templates, we design a chain-of-thought method for large language models (LLMs). LLMs exhibit exceptional semantic comprehension, deftly distinguishing between parameters and invariant tokens. We have conducted experiments on large-scale public datasets. Extensive evaluation
demonstrates that (<img src="./assets/Lemur-emo.png" width="3%">LEMUR)
achieves the state-of-the-art performance and impressive efficiency.
