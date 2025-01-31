from rich.tree import Tree
from rich.console import Console
import os

def build_tree(directory, tree):
    # Filter out hidden files and directories
    for item in os.listdir(directory):
        if item.startswith("."):
            continue  # Skip hidden files and directories
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            branch = tree.add(f"[bold blue]{item}/[/]")  # Add directories in blue
            build_tree(item_path, branch)
        else:
            tree.add(f"{item}")  # Add files

def display_directory_tree(directory):
    console = Console()
    tree = Tree(f"[bold green]{os.path.basename(directory)}/[/]")  # Root directory in green
    build_tree(directory, tree)
    console.print(tree)

# 调用函数
display_directory_tree('/home/whx/FD-Bench')