# ----------------------------------------------------------------------------------
# - Author Contact: wei.zhang, zwpride@buaa.edu.cn (Original code)
# ----------------------------------------------------------------------------------

from collections import Counter
from collections import defaultdict
import math

class Vectorizer:
    def __init__(self):
        self.documents = {}
        self.common_subsequences = None
        self.difference_from_common_subsequences = {}
        self.token_stats_at_position = {}
        self.tokens_stats_at_position_all_docs = defaultdict(list)

    def fit(self, documents):
        self.documents = documents
        # self.get_longest_common_subsequences()
        return self

    def _longest_common_subsequence(self, sequence1, sequence2):
        lengths = [
            [0 for j in range(len(sequence2) + 1)] for i in range(len(sequence1) + 1)
        ]
        for i, x in enumerate(sequence1):
            for j, y in enumerate(sequence2):
                if x == y:
                    lengths[i + 1][j + 1] = lengths[i][j] + 1
                else:
                    lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
        result = []
        x, y = len(sequence1), len(sequence2)
        while x != 0 and y != 0:
            if lengths[x][y] == lengths[x - 1][y]:
                x -= 1
            elif lengths[x][y] == lengths[x][y - 1]:
                y -= 1
            elif sequence1[x - 1] == sequence2[y - 1]:
                result.insert(0, sequence1[x - 1])
                x -= 1
                y -= 1
        return result

    def get_longest_common_subsequences(self):
        if self.common_subsequences is None:
            sequences = list(self.documents.values())
            common_subsequences = sequences[0]
            for seq in sequences[1:]:
                common_subsequences = self._longest_common_subsequence(
                    common_subsequences, seq
                )
            self.common_subsequences = common_subsequences
        return self.common_subsequences

    def get_difference_from_common_subsequences(self, doc_id):
        if doc_id not in self.documents:
            return None, None
        elif doc_id not in self.difference_from_common_subsequences:
            common_subsequences = self.get_longest_common_subsequences()
            difference = [
                (token, pos)
                for pos, token in enumerate(self.documents[doc_id])
                if token not in common_subsequences
            ]
            self.difference_from_common_subsequences[doc_id] = (difference)
        return self.difference_from_common_subsequences[doc_id]

    def get_token_stats_at_position(self, doc_id, position):
        if doc_id not in self.documents or len(self.documents[doc_id]) <= position:
            return None
        elif (doc_id, position) not in self.token_stats_at_position:
            token = self.documents[doc_id][position]
            tokens_in_same_position_other_docs = Counter(
                doc[position] for doc in self.documents.values() if len(doc) > position
            )
            freq = tokens_in_same_position_other_docs[token]
            freq_percent = freq / len(self.documents)
            var_types = len(tokens_in_same_position_other_docs)
            self.token_stats_at_position[(doc_id, position)] = (token, freq, freq_percent, var_types)
        return self.token_stats_at_position[(doc_id, position)]

    def calculate_entropy_at_position(self, position):
        stats = self.get_tokens_stats_at_position_all_docs(position)
        total_docs = len(self.documents)
        entropy = 0
        for token, freq, _ in stats:
            prob = freq / total_docs
            entropy -= prob * math.log2(prob)
        return entropy, stats
    
    def get_tokens_stats_at_position_all_docs(self, position):
        if not self.tokens_stats_at_position_all_docs[position]:
            tokens_in_pos = [
                doc[position] for doc in self.documents.values() if len(doc) > position
            ]
            freq_map = Counter(tokens_in_pos)
            total_docs = len(self.documents)
            stats = [
                (token, freq, freq / total_docs) for token, freq in freq_map.items()
            ]
            self.tokens_stats_at_position_all_docs[position] = stats
        return self.tokens_stats_at_position_all_docs[position]

def main():
    documents = {
        0: [
            "proxy.cse.cuhkedu.hk:5070",
            "open",
            "through",
            "proxy",
            "proxy.cse.cuhk.edu.hk:5070",
            "HTTPS",
        ],
        # 1: ["proxy.cse.cuhkedu.hk:5070", "close", "0", "bytes", "sent", "0", "bytes", "received", "lifetime", "00:01"],
        2: [
            "proxy.cse.cuhkedu.hk:5070",
            "open",
            "through",
            "proxy",
            "p3p.sogou.com:80",
            "HTTPS",
        ],
        3: [
            "proxy.cse.cuhkedu.hk:5070",
            "open",
            "through",
            "proxy",
            "182.254.114.110:80",
            "SOCKS5",
        ],
        4: [
            "182.254.114.110:80",
            "open",
            "through",
            "proxy",
            "182.254.114.110:80",
            "HTTPS",
        ],
        5: ["proxy.cse.cuhkedu.hk:5070", "close", "403", "bytes", "sent", "426", "bytes", "received", "lifetime", "00:02"],
        # 6: ["get.sogou.com:80", "close", "651", "bytes", "sent", "346", "bytes", "received", "lifetime", "00:03"],
        # 7: ["proxy.cse.cuhkedu.hk:5070", "close", "408", "bytes", "sent", "421", "bytes", "received", "lifetime", "00:03"],
        8: [
            "183.62.156.108:22",
            "open",
            "through",
            "proxy",
            "proxy.cse.cuhk.edu.hk:5070",
            "SOCKS5",
        ],
        9: [
            "proxy.cse.cuhkedu.hk:5070",
            "open",
            "through",
            "proxy",
            "proxy.cse.cuhk.edu.hk:5070",
            "SOCKS5",
        ],
        10: [
            "proxy.cse.cuhkedu.hk:5070",
            "open",
            "through",
            "proxy",
            "socks.cse.cuhk.edu.hk:5070",
            "HTTPS",
        ],
        11: ["open", "open", "through", "proxy", "socks.cse.cuhk.edu.hk:5070", "proxy"],
    }
    vectorizer = Vectorizer().fit(documents)

    # Testing get_longest_common_subsequences method
    print("Longest common subsequences:")
    print(vectorizer.get_longest_common_subsequences())

    # Testing get_difference_from_common_subsequences method for doc_id=2
    print("\nDifference and Common sequences for document id 11:")
    diff = vectorizer.get_difference_from_common_subsequences(11)
    # print("Common:", common)
    print("Difference:", diff)

    did, pos = 11, 0
    # Testing get_token_stats_at_position method for doc_id=2 and position=1
    print(f"\nToken stats at position {pos} for document id {did}:")
    token, freq, freq_percent, var_types = vectorizer.get_token_stats_at_position(did, pos)
    print(
        f"Token: {token}, Frequency: {freq}, Frequency percent: {freq_percent}, Var types: {var_types}"
    )


    position = 0
    stats = vectorizer.get_tokens_stats_at_position_all_docs(position)
    print(f"\nToken stats at position {position} for all documents:")
    for s in stats:
        print(f"Token: {s[0]}, Frequency: {s[1]} Frequency percent: {s[2]}")
    print(f"Token types at position {position}: {len(stats)}")


if __name__ == "__main__":
    main()
