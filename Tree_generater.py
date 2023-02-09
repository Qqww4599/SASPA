import os
import argparse

'''
建立文件目錄用腳本。來自https://blog.csdn.net/qq_36102055/article/details/117690302，謝謝。
'''


class TreeNode:
    def __init__(self, name: str = None, child: list = None) -> None:
        self.child = child
        self.name = name


class TreeGenerator:

    @classmethod
    def getTreeStructure(cls, path: str, now: TreeNode) -> list:
        now.name = os.path.basename(path)
        if not os.path.isdir(path):
            return ['This is not a folder']
        else:
            now.child = []
            for son in [os.path.join(path, name) for name in os.listdir(path)]:
                ntree = TreeNode()
                now.child.append(ntree)
                cls.getTreeStructure(son, ntree)

    @classmethod
    def generateGraph(cls, path: list) -> str:
        """
        给定一个文件夹路径，生成对应的树形结构文本
        """
        graph = ''
        batch, horizontal, vertical, sep, end = '┣', '━', '┃', '    ', '┗'

        root = TreeNode()
        cls.getTreeStructure(path, root)

        def generator(node: TreeNode, back: str, isroot: str):
            nonlocal graph
            if len(back):
                graph += back[5:] + node.name + '\n'
            else:
                graph += back + node.name + '\n'

            if node.child is not None:
                for x in range(len(node.child)):
                    if x == len(node.child) - 1:
                        node.child[x].name = sep + end + horizontal * 2 + node.child[x].name
                        generator(node.child[x], back + sep + isroot, ' ')
                    else:
                        node.child[x].name = sep + batch + horizontal * 2 + node.child[x].name
                        generator(node.child[x], back + sep + isroot, vertical)

        generator(root, '', vertical)
        print(graph)
        return graph

    @classmethod
    def save(cls, tree_graph: str, save_path: str) -> None:
        """
        保存树形结构文本
        """

        with open(save_path, encoding='utf-8', mode='w') as w:
            w.write(tree_graph)


if __name__ == '__main__':
    args = argparse.ArgumentParser('Parameters')
    args.add_argument('--recordPath', default=True, type=str)
    args = args.parse_args()
    TreeGenerator.save(TreeGenerator.generateGraph(args.recordPath), './tree.txt')