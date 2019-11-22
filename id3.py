import numpy as np
import pandas as pd
from math import log2
from anytree import Node, RenderTree

class ID3():
    def __init__(self, dat_filename):
        self.orig_df = pd.read_csv(dat_filename)
        self.df = self.orig_df
        self.num_rows = self.df.shape[0]
        self.data_map = {}
        self.uv_map = {}
        self.tree = None
        self.generate_datamap()
        self.labels = list(self.data_map)
        # print(self.labels)
        self.generate_uvmap()

    # Generate a Data Map from the Dataframe
    def generate_datamap(self, subset = pd.DataFrame()):
        df = None
        if subset.empty:
            df = self.df
        else:
            df = subset

        columns = list(df.columns)
        data_shape = df.shape
        # print(self.data)
        self.data_map = {}
        for c in columns:
            # print(c)
            col_dat = df.loc[:data_shape[0], c]
            self.data_map.update({c: col_dat.tolist()})
        # print(self.data_map)

    # Generate a Unique Values Map from the Data Map
    def generate_uvmap(self):
        self.uv_map = {}
        for key in self.data_map:
            # print(key)
            aggregator_obj = {}
            for val in self.data_map[key]:
                # print(val)
                if val in aggregator_obj:
                    aggregator_obj[val] += 1
                else:
                    # add new value to map
                    aggregator_obj.update({val: 1})
            self.uv_map.update({key: aggregator_obj})
        # print(self.uv_map)

    def generate_ig_calculation_data(self, target, key):
        pass

    def get_label_values(self, label):
        data = self.uv_map[label]
        return list(data)

    def entropy(self, keyword, tab=0):
        tabs = self.generate_tabs(tab)
        entropy_val = 0
        work_str = 'E(S)='

        dat = (self.uv_map[keyword])
        keys = list(dat)
        total = self.df.shape[0]

        for k in keys:
            # Calculate part of entropy
            prob = dat[k]/total
            print(tabs + 'p = ' + str(dat[k]) + '/' + str(total))
            # print(prob)
            entropy_val -= (prob*log2(prob))
            work_str += ' - ('+str(prob)+')log2('+str(prob)+')'
        print(tabs + work_str)
        return entropy_val

    def calc_entropy(self, prob_list, tab=0):
        tabs = self.generate_tabs(tab)
        work_str = 'E(S)='
        entropy_val = 0
        for p in prob_list:
            if(p > 0):
                entropy_val -= (p*log2(p))
                work_str += ' - (' + str(p) + 'log2(' + str(p) + ')'
            else:
                print(tabs + 'ignoring 0 probability')
        print(tabs + work_str)
        return entropy_val

    def print_entropy(self, k, val, tab=0):
        tabs = self.generate_tabs(tab)
        print(tabs + "E(S_" + k + ")= " + str(val))

    def get_subset_of_feature(self, feature, feat_val, target, tab=0):
        tabs = self.generate_tabs(tab)
        target_vals = self.get_label_values(target)
        subset = []

        select_frame = self.df[[feature, target]][self.df[feature] == feat_val]
        # print(select_frame)
        total = select_frame.shape[0]
        print(tabs + 'total for ' + feature + ', ' + str(feat_val) + ': ' + str(total))

        # get subset with feature and target
        for v in target_vals:
            select_frame = self.df[[feature, target]][(self.df[feature] == feat_val) & (self.df[target] == v)]
            # print(select_frame)
            num_feat_val = select_frame.shape[0]
            print(tabs + 'subset for ' + str(feat_val) + ', ' + str(v) + ': ' + str(num_feat_val) + '/' + str(total) + ' = ' + str(num_feat_val/total))
            subset.append((num_feat_val/total))
        return subset


    def info_gain(self, feature, target, target_entropy, tab=0):
        feature_entropies = []
        # 1. Get Probabilities
        print(self.uv_map[feature])
        prob_data = self.uv_map[feature]
        # print(prob_data)
        prob_keys = list(prob_data)
        prob_map = {}
        prob_list = []
        for l in prob_keys:
            prob = prob_data[l]/self.num_rows
            # prob_label = "probability for " + feature + ', ' + l
            # self.print_with_tab(prob, prob_label, tab )
            prob_map.update({l: prob})
            prob_list.append(prob)
        self.print_with_tab(prob_map, feature + " probabilities: ", tab)

        # 2. Get Entropy of different keys wrt target
        for l in prob_keys:
            label = feature + ', ' + str(l)
            print(self.generate_tabs(tab) + "E(S_" + label + ") ")
            probability_list = self.get_subset_of_feature(feature, l, target, tab+1)
            # print(probability_list)
            entropy_k = self.calc_entropy(probability_list, tab+1)
            self.print_entropy(label, entropy_k, tab)
            feature_entropies.append(entropy_k)

        # 3. E(target) - Summation of feature entropies wrt target
        ig = target_entropy
        work_str = str(target_entropy)
        for i in range(len(feature_entropies)):
            ig -= feature_entropies[i]*prob_list[i]
            work_str += ' - (' + str(prob_list[i]) + ')' + str(feature_entropies[i])
        self.print_info_gain(target, feature, work_str, tab)
        return ig

    def print_info_gain(self, parent, k, val, tab=0):
        tabs = self.generate_tabs(tab)
        print(tabs + "IG(S_" + parent + ", " + k + ")= " + str(val))

    def print_with_tab(self, data, label, tab=0):
        tabs = self.generate_tabs(tab)
        print(tabs + label + ": " + str(data))

    def generate_tabs(self, tabs):
        tab_str = ''
        for i in range(tabs):
            tab_str += '\t'
        return tab_str

    def print_map(self, map, pre, tab=0):
        tabs = self.generate_tabs(tab)
        print(tabs + pre + ':')
        for k in map:
            print(self.generate_tabs(tab+1) + str(k) + ': '+ str(map[k]))

    def print_tree(self):
        if self.tree != None:
            print('\n\n------------ R E S U L T I N G  T R E E ------------')
            print(RenderTree(self.tree))
        else:
            print('tree is rootless')

    # ID3 Algorithm
    #   1. Calculate entropy of each attribute of the data set S
    #   2. Partition ("split") the set, S into subsets using the attribute for which the resulting entropy after splitting is minimized; or, equivalently, information gain is maximum.
    #   3. Make a decision tree node containing that attribute.
    #   4. Perform the steps recursively on subsets using the remaining attributes, adding branches and connecting nodes.
    def performID3(self, layer=0, target = None, feature_val=None, title='', parent_set=pd.DataFrame(), child_branch=0, parent_node=None, subset_labels=None):
        tab = layer + 1
        tabs = self.generate_tabs(tab)
        curr_node = None

        # Get entropy of overall set
        if subset_labels == None:
            length = len(self.labels)
        else:
            length = len(subset_labels)
        # print(length)
        # print(self.labels[length-1])
        if target == None:
            # set target to target output feature
            target = self.labels[length-1]
        if layer==0:
            print('\n' + tabs + '---------- Performing ID3 for: ' + target + ', Layer = ' + str(layer) + ' ----------\n')
        else:
            print('\n' + tabs + '---------- Performing ID3 for: ' + title + ', Layer = ' + str(layer) + ' ----------\n')
        ig_map = {}
        # print(key)

        entropy_of_target = self.entropy(target, tab)
        self.print_entropy(title, entropy_of_target, tab)
        if entropy_of_target <= 0:
            # no information gain
            print(tabs + 'No information to gain from this set')
        else:
            print(tabs + 'Entropy of ' + target + ' indicates information content can be gained ')
            if(length-1 == 0):
                for i in range(1):
                    # Calc Info Gain
                    if subset_labels == None:
                        feature = self.labels[i]
                    else:
                        feature = subset_labels[i]
                    print('\n' + tabs + 'IG for ' + feature)
                    ig = self.info_gain(feature, target, entropy_of_target, tab)
                    self.print_info_gain(target, feature, ig, tab)
                    ig_map.update({feature: ig})
            else:
                for i in range(length-1):
                    # Calc Info Gain
                    if subset_labels == None:
                        feature = self.labels[i]
                    else:
                        feature = subset_labels[i]
                    print('\n' + tabs + 'IG for ' + feature)
                    ig = self.info_gain(feature, target, entropy_of_target, tab)
                    self.print_info_gain(target, feature, ig, tab)
                    ig_map.update({feature: ig})
        print('')

        if entropy_of_target > 0:
            self.print_map(ig_map, "Info Gains", tab)
            max_val = max(ig_map, key=ig_map.get)
            print(tabs + 'Greatest IG is ' + max_val)
        else:
            max_val = feature_val

        # add to tree
        if layer == 0:
            if self.tree is None:
                print(tabs + 'Adding ' + str(max_val) + ' as root')
                # self.tree.update({max_val: Node})
                self.tree = Node(max_val)
                curr_node = self.tree
                self.print_tree()
        else:
            if entropy_of_target > 0:
                print(tabs + 'Adding ' + str(max_val) + ' to tree')
                curr_node = Node(max_val, parent=parent_node)
                self.print_tree()

            # self.tree.children[child_branch].add_child(layer+1, Node(max_val))

        # Call Perform ID3 for children
        if entropy_of_target > 0:
            child_labels = self.get_label_values(max_val)
            possible_children = len(self.get_label_values(max_val))
            print('possible Children' + str(child_labels))
            for i in range(possible_children):
                print('\n\n' + tabs + '----- Creating subset for branch ' + str(child_labels[i]) + ' -----')
                if parent_set.empty:
                    self.df = self.orig_df
                else:
                    self.df = parent_set
                    # remove column of parent data
                self.df = self.df[self.df[max_val] == child_labels[i]]
                self.df.reset_index(inplace=True, drop=True)
                # print('before')
                print(self.df)
                self.generate_datamap(self.df)
                self.generate_uvmap()
                self.df = self.df.drop([max_val], axis=1)
                # print('after')
                # print(self.df)
                # print('list: ' + str(list(self.data_map)))
                newLabels = list(self.data_map)
                # print(str(max_val))
                # print(newLabels.index(max_val))
                newLabels.remove(max_val)
                print('Labels to use: ' + str(newLabels))
                temp = Node(str(child_labels[i]), parent=curr_node)
                self.performID3(layer+1, target, feature_val=child_labels[i], title=max_val+'_'+str(child_labels[i]), parent_set= self.df, child_branch=i, parent_node=temp, subset_labels=newLabels)
        else:
            # print(list(self.uv_map[target])[0])
            print(tabs + 'leaf node \'' + str(list(self.uv_map[target])[0]) + '\' added to branch ' + str(max_val))
            curr_node = Node(str(list(self.uv_map[target])[0]), parent=parent_node)
            self.print_tree()


# M A I N

# DATA PROCESSING
id3 = ID3('data.csv')
# id3 = ID3('tennis.csv')
# id3 = ID3('random_data.csv')
# id3 = ID3('processed.cleveland.csv')
# id3 = ID3('titanic.csv')
id3.performID3()