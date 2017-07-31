# -*- coding:utf-8 -*-
# author: jrlimingyang@jd.com

from collections import defaultdict, namedtuple
from itertools import imap
import pandas as pd
from optparse import OptionParser
import os 

VERBOSE = False # verbose output

def fpgrowth(dataset, min_support, include_support=False):
    F = []
    support_data = {}
    for k,v in find_frequent_itemsets(dataset, min_support, include_support=True):
        F.append(frozenset(k))
        support_data[frozenset(k)] = v

    # Create one array with subarrays that hold all transactions of equal length.
    def bucket_list(nested_list, sort=True):
        bucket = defaultdict(list)
        for sublist in nested_list:
            bucket[len(sublist)].append(sublist)
        return [v for k,v in sorted(bucket.items())] if sort else bucket.values()

    F = bucket_list(F)
    return F, support_data

def find_frequent_itemsets(dataset, min_support, include_support=False):
    items = defaultdict(lambda: 0) # mapping from items to their supports
    processed_transactions = []

    # Load the passed-in transactions and count the support that individual
    # items have.
    for transaction in dataset:
        processed = []
        for item in transaction:
            items[item] += 1
            processed.append(item)
        processed_transactions.append(processed)

    # Remove infrequent items from the item support dictionary.
    items = dict((item, support) for item, support in items.iteritems()
        if support >= min_support)

    # Build our FP-tree. Before any transactions can be added to the tree, they
    # must be stripped of infrequent items and their surviving items must be
    # sorted in decreasing order of frequency.
    def clean_transaction(transaction):
        transaction = filter(lambda v: v in items, transaction)
        transaction.sort(key=lambda v: items[v], reverse=True)
        return transaction

    master = FPTree()
    for transaction in imap(clean_transaction, processed_transactions):
        master.add(transaction)

    support_data = {}
    def find_with_suffix(tree, suffix):
        for item, nodes in tree.items():
            support = float(sum(n.count for n in nodes)) / len(dataset)
            if support >= min_support and item not in suffix:
                # New winner!
                found_set = [item] + suffix
                support_data[frozenset(found_set)] = support
                yield (found_set, support) if include_support else found_set

                # Build a conditional tree and recursively search for frequent
                # itemsets within it.
                cond_tree = conditional_tree_from_paths(tree.prefix_paths(item),
                    min_support)
                for s in find_with_suffix(cond_tree, found_set):
                    yield s # pass along the good news to our caller

    if VERBOSE:
        # Print a list of all the frequent itemsets.
        for itemset, support in find_with_suffix(master, []):
            print "" \
                + "{" \
                + "".join(str(i) + ", " for i in iter(itemset)).rstrip(', ') \
                + "}" \
                + ":  sup = " + str(round(support_data[frozenset(itemset)], 3))

    # Search for frequent itemsets, and yield the results we find.
    for itemset in find_with_suffix(master, []):
        yield itemset

class FPTree(object):
    Route = namedtuple('Route', 'head tail')

    def __init__(self):
        # The root node of the tree.
        self._root = FPNode(self, None, None)

        # A dictionary mapping items to the head and tail of a path of
        # "neighbors" that will hit every node containing that item.
        self._routes = {}

    @property
    def root(self):
        """The root node of the tree."""
        return self._root

    def add(self, transaction):
        """
        Adds a transaction to the tree.
        """

        point = self._root

        for item in transaction:
            next_point = point.search(item)
            if next_point:
                # There is already a node in this tree for the current
                # transaction item; reuse it.
                next_point.increment()
            else:
                # Create a new point and add it as a child of the point we're
                # currently looking at.
                next_point = FPNode(self, item)
                point.add(next_point)

                # Update the route of nodes that contain this item to include
                # our new node.
                self._update_route(next_point)

            point = next_point

    def _update_route(self, point):
        """Add the given node to the route through all nodes for its item."""
        assert self is point.tree

        try:
            route = self._routes[point.item]
            route[1].neighbor = point # route[1] is the tail
            self._routes[point.item] = self.Route(route[0], point)
        except KeyError:
            # First node for this item; start a new route.
            self._routes[point.item] = self.Route(point, point)

    def items(self):
        """
        Generate one 2-tuples for each item represented in the tree. The first
        element of the tuple is the item itself, and the second element is a
        generator that will yield the nodes in the tree that belong to the item.
        """
        for item in self._routes.iterkeys():
            yield (item, self.nodes(item))

    def nodes(self, item):
        """
        Generates the sequence of nodes that contain the given item.
        """

        try:
            node = self._routes[item][0]
        except KeyError:
            return

        while node:
            yield node
            node = node.neighbor

    def prefix_paths(self, item):
        """Generates the prefix paths that end with the given item."""

        def collect_path(node):
            path = []
            while node and not node.root:
                path.append(node)
                node = node.parent
            path.reverse()
            return path

        return (collect_path(node) for node in self.nodes(item))

    def inspect(self):
        print 'Tree:'
        self.root.inspect(1)

        print
        print 'Routes:'
        for item, nodes in self.items():
            print '  %r' % item
            for node in nodes:
                print '    %r' % node

    def _removed(self, node):
        """Called when `node` is removed from the tree; performs cleanup."""

        head, tail = self._routes[node.item]
        if node is head:
            if node is tail or not node.neighbor:
                # It was the sole node.
                del self._routes[node.item]
            else:
                self._routes[node.item] = self.Route(node.neighbor, tail)
        else:
            for n in self.nodes(node.item):
                if n.neighbor is node:
                    n.neighbor = node.neighbor # skip over
                    if node is tail:
                        self._routes[node.item] = self.Route(head, n)
                    break

def conditional_tree_from_paths(paths, min_support):
    """Builds a conditional FP-tree from the given prefix paths."""
    tree = FPTree()
    condition_item = None
    items = set()

    # Import the nodes in the paths into the new tree. Only the counts of the
    # leaf notes matter; the remaining counts will be reconstructed from the
    # leaf counts.
    for path in paths:
        if condition_item is None:
            condition_item = path[-1].item

        point = tree.root
        for node in path:
            next_point = point.search(node.item)
            if not next_point:
                # Add a new node to the tree.
                items.add(node.item)
                count = node.count if node.item == condition_item else 0
                next_point = FPNode(tree, node.item, count)
                point.add(next_point)
                tree._update_route(next_point)
            point = next_point

    assert condition_item is not None

    # Calculate the counts of the non-leaf nodes.
    for path in tree.prefix_paths(condition_item):
        count = path[-1].count
        for node in reversed(path[:-1]):
            node._count += count

    # Eliminate the nodes for any items that are no longer frequent.
    for item in items:
        support = sum(n.count for n in tree.nodes(item))
        if support < min_support:
            # Doesn't make the cut anymore
            for node in tree.nodes(item):
                if node.parent is not None:
                    node.parent.remove(node)

    # Finally, remove the nodes corresponding to the item for which this
    # conditional tree was generated.
    for node in tree.nodes(condition_item):
        if node.parent is not None: # the node might already be an orphan
            node.parent.remove(node)

    return tree

class FPNode(object):
    """A node in an FP tree."""

    def __init__(self, tree, item, count=1):
        self._tree = tree
        self._item = item
        self._count = count
        self._parent = None
        self._children = {}
        self._neighbor = None

    def add(self, child):
        """Adds the given FPNode `child` as a child of this node."""

        if not isinstance(child, FPNode):
            raise TypeError("Can only add other FPNodes as children")

        if not child.item in self._children:
            self._children[child.item] = child
            child.parent = self

    def search(self, item):
        """
        Checks to see if this node contains a child node for the given item.
        If so, that node is returned; otherwise, `None` is returned.
        """

        try:
            return self._children[item]
        except KeyError:
            return None

    def remove(self, child):
        try:
            if self._children[child.item] is child:
                del self._children[child.item]
                child.parent = None
                self._tree._removed(child)
                for sub_child in child.children:
                    try:
                        # Merger case: we already have a child for that item, so
                        # add the sub-child's count to our child's count.
                        self._children[sub_child.item]._count += sub_child.count
                        sub_child.parent = None # it's an orphan now
                    except KeyError:
                        # Turns out we don't actually have a child, so just add
                        # the sub-child as our own child.
                        self.add(sub_child)
                child._children = {}
            else:
                raise ValueError("that node is not a child of this node")
        except KeyError:
            raise ValueError("that node is not a child of this node")

    def __contains__(self, item):
        return item in self._children

    @property
    def tree(self):
        """The tree in which this node appears."""
        return self._tree

    @property
    def item(self):
        """The item contained in this node."""
        return self._item

    @property
    def count(self):
        """The count associated with this node's item."""
        return self._count

    def increment(self):
        """Increments the count associated with this node's item."""
        if self._count is None:
            raise ValueError("Root nodes have no associated count.")
        self._count += 1

    @property
    def root(self):
        """True if this node is the root of a tree; false if otherwise."""
        return self._item is None and self._count is None

    @property
    def leaf(self):
        """True if this node is a leaf in the tree; false if otherwise."""
        return len(self._children) == 0

    def parent():
        doc = "The node's parent."
        def fget(self):
            return self._parent
        def fset(self, value):
            if value is not None and not isinstance(value, FPNode):
                raise TypeError("A node must have an FPNode as a parent.")
            if value and value.tree is not self.tree:
                raise ValueError("Cannot have a parent from another tree.")
            self._parent = value
        return locals()
    parent = property(**parent())

    def neighbor():
        doc = """
        The node's neighbor; the one with the same value that is "to the right"
        of it in the tree.
        """
        def fget(self):
            return self._neighbor
        def fset(self, value):
            if value is not None and not isinstance(value, FPNode):
                raise TypeError("A node must have an FPNode as a neighbor.")
            if value and value.tree is not self.tree:
                raise ValueError("Cannot have a neighbor from another tree.")
            self._neighbor = value
        return locals()
    neighbor = property(**neighbor())

    @property
    def children(self):
        """The nodes that are children of this node."""
        return tuple(self._children.itervalues())

    def inspect(self, depth=0):
        print ('  ' * depth) + repr(self)
        for child in self.children:
            child.inspect(depth + 1)

    def __repr__(self):
        if self.root:
            return "<%s (root)>" % type(self).__name__
        return "<%s %r (%r)>" % (type(self).__name__, self.item, self.count)

def get_input(parser):
    '''parse the arguments provided in the commond line'''
    parser.add_option('-p', '--filepath', action='store', type='string', help='input filepath', dest='filepath')
    parser.add_option('-n', '--filename', action='store', type='string', help='input filename', dest='filename' )
    parser.add_option('-t', '--target', action='store', type='string', help='target column', dest='target')
    parser.add_option('-s', '--support', action='store', type='float', help='min support', dest='support')
    parser.add_option('-g', '--group', action='store', type='int', help='group size', dest='group')
    parser.add_option('-r', '--rate', action='store', type='float', help='bad rate', dest='rate')

    (options, args) = parser.parse_args()
    if not options.filepath:
        print 'Input filepath not given, please input ...'
    if not options.filename:
        print 'Input filename not given, please input ...'
    if not options.target:
        print 'Target column not given, please input ...'
    if not options.support:
        print 'Support not given, will use default(0.1).'
    if not options.group:
        print 'Group size not given, will use default(50).'
    if not options.rate:
        print 'Bad rate not given, will use default(0.5)'

    return (options, args)



def load_dataset(filefath,filename):
    raw_data = pd.read_csv(filepath+filename)
    raw_data.iloc[:,0:-1].to_csv(filepath+'temp.csv')
    dataset = []
    for i, line in enumerate(open(filepath+'temp.csv').readlines()):
        if i > 0:
            lineset = []
            for item in line.split(','):
                lineset.append(item.strip())
            #lineset.append(line.strip())
            dataset.append(lineset)
    return raw_data,dataset

def _pretty_print(options):
    print '='*20
    print ''
    print '*Options'
    print '  -p filepath:%s' %options.filepath
    print '  -n filename:%s' %options.filename
    print '  -t target column:%s' %options.target
    print '  -s support:%s' %options.support
    print '  -g group size:%s' %options.group
    print '  -r bad rate:%s' %options.rate
    print ''
    print '='*20

def pprint_frequent(F,group_number,min_bad_rate, target,raw_data):
    print 'Start outputing results...'
    frequet_set = {}
    l = len(F)
    index = 0
    df = raw_data.copy()
    print 'Frequent Itemsets',' '*4,'Support',' '*4,'Group Size',' '*4,'Bad Customer Num',' '*4,'Bad Rate'
    for itemset in F:
        total_num,_ = df.shape
        index += 1
        print ''
        print '-'*20
        print '{0} Item(s) Frequent sets'.format(index)
        print '-'*20
        if index == 1:
            for item in list(itemset):
                key = list(item)[0].split(':')[0]
                value = list(item)[0].split(':')[1]
                temp = ('('+'df["{0}"]=="{1}"'+')').format(key,list(item)[0])
                df1 = eval('df[{0}]'.format(temp))
                fenzi,_ = df1[df1[target]==target +':1'].shape
                fenmu,_ = df1.shape
                support = float(fenmu)/total_num
                
                rates = float(fenzi)/fenmu
                if rates > min_bad_rate:
                    if fenmu > group_number:
                        print list(item),'-*'*4,round(support,3),'-*'*4,fenmu,'-*'*4,fenzi,'-*'*4,round(rates,3)
        else:
            for item in list(itemset):
                string = ''
                flag = 0
                
                for itemsub in list(item):
                    n = len(list(item))
                    flag += 1
                    key = itemsub.split(':')[0]
                    value = itemsub.split(':')[1]
                    temp = ('('+'df["{0}"]=="{1}"'+')').format(key,itemsub)

                    if flag < n:
                        temp += '&'
                    else:
                        temp = temp
                    string += temp
                df1 = eval('df[{0}]'.format(string))
                fenzi,_ = df1[df1['overdue']=='overdue:1'].shape
                fenmu,_ = df1.shape
                support = float(fenmu)/total_num
                rates = float(fenzi)/fenmu
                if (rates > min_bad_rate):
                    if (fenmu > group_number):
                        print list(item),'-*'*4,round(support,3),'-*'*4,fenmu,'-*'*4,fenzi,'-*'*4,round(rates,3)
    print 'Complete print!'


if __name__ == '__main__':   
    print 'Sniffer Model Start!'

    usage_text = 'Usage: %prog[options] -p filepath -n filename -t target_col -s minsup -g groupsz -r badrate'
    
    parser = OptionParser(usage=usage_text)
    (options, args) = get_input(parser)
    if options.filepath is not None:
        filepath = options.filepath
    if options.filename is not None:
        filename = options.filename
    if options.target is not None:
        target = options.target
    
    raw_data, dataset = load_dataset(filepath,filename)

     # list of transactions; each transaction is a list of items
    D = map(set, dataset) # set of transactions; each transaction is a list of items

    # Generate all the frequent itemsets using the FP-growth algorithm.
    if options.support is not None:
        min_support = options.support
    else:
        min_support = 0.1

    F, support_data = fpgrowth(dataset, min_support)

    if options.group is not None:
        group = options.group
    else:
        group = 50
 
    if options.rate is not None:
        rate = options.rate
    else:
        rate = 0.2

    _pretty_print(options)
    pprint_frequent(F, group, rate, target, raw_data)

