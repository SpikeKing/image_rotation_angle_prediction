class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        res = []
        a_list = ["a", "e", "i", "o", "u"]
        tmp_list = []
        tmp = 0
        for c in s:
            if len(tmp_list) == k:
                if tmp_list[0] in a_list:
                    tmp -= 1
                tmp_list = tmp_list[1:]
                if c in a_list:
                    tmp += 1
                tmp_list.append(c)
            else:
                tmp_list.append(c)
                if c in a_list:
                    tmp += 1
            if tmp == k:
                return k
            print(tmp_list, tmp, res)
            res.append(tmp)
        return max(res) if res else 0


def main():
    so = Solution()
    r = so.maxVowels("weallloveyou", 7)
    print(r)


if __name__ == '__main__':
    main()