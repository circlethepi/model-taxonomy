
def sort_taxonomy(values):
    # values = [x.replace("dataset_embedding", "dataset") for x in values]
    order = ["recipe", "dataset_embedding", "structural", "functional", "behavioral"]
    return sorted(values, key=lambda x: order.index(x) if x in order else len(order))