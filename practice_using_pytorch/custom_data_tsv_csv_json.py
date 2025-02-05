from torchtext.data import TabularDataset, Field, BucketIterator

tokenize = lambda x: x.split()

quote = Field(sequential=True, use_vocab = True, tokenize=tokenize, lower=True)
score = Field(sequential=False, use_vocab = False)

fields = {"quote":("q", quote), "score":("s", score)}

train_data, test_data = TabularDataset.split(
    path = "mydata",
    train = "train.json",
    test = "test.json",
    #validation = "validation.json"
    format = "json",
    fileds = fields
)

# train_data, test_data = TabularDataset.split(
#     path = "mydata",
#     train = "train.csv",
#     test = "test.csv",
#     format = "csv",
#     fileds = fields
# )

# train_data, test_data = TabularDataset.split(
#     path = "mydata",
#     train = "train.tsv",
#     test = "test.tsv",
#     format = "tsv",
#     fileds = fields
# )

# print(train_data[0].__dict__.keys())
# print(train_data[0].__dict__.values())

quote.build_vocab(train_data,
                  max_size = 10000,
                  min_freq = 1)

train_iterator, test_iterator = BucketIterator.split(
    (train_data, test_data),
    batch_size=2,
    device = "cuda"
)