# !/usr/bin/env python
import sys
import os
import os.path
import json

from evaluation import truth_file_name
from utils import Accuracy


def gen_train_facts(data_file_name, truth_dir):
	fact_file_name = data_file_name[data_file_name.find("train_"):]
	fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

	if os.path.exists(fact_file_name):
		fact_in_train = set([])
		triples = json.load(open(fact_file_name))
		for x in triples:
			fact_in_train.add(tuple(x))
		return fact_in_train

	fact_in_train = set([])
	ori_data = json.load(open(data_file_name))
	for data in ori_data:
		vertexSet = data['vertexSet']
		for label in data['labels']:
			rel = label['r']
			for n1 in vertexSet[label['h']]:
				for n2 in vertexSet[label['t']]:
					fact_in_train.add((n1['name'], n2['name'], rel))

	json.dump(list(fact_in_train), open(fact_file_name, "w"))

	return fact_in_train

def evaluate_by_rel(submission_answer, truth, fact_in_train_annotated, fact_in_train_distant, use_wikidataid = True):
	print("Warning: different definition of Ign")
	rel2id = json.load(open(os.path.join("../src/prepro_data", 'rel2id.json')))
	id2rel = {v: k for k, v in rel2id.items()}

	accu_re = Accuracy()
	accu_re_ign_train_annotated = Accuracy()
	accu_re_ign_train_distant = Accuracy()
	accu_evi = Accuracy()

	accu_re_ign_train_annotated_by_rel = [Accuracy() for _ in range(len(rel2id))]
	accu_re_ign_train_distant_by_rel = [Accuracy() for _ in range(len(rel2id))]
	accu_re_by_rel = [Accuracy() for _ in range(len(rel2id))]
	accu_evi_by_rel = [Accuracy() for _ in range(len(rel2id))]


	std = {}
	std_ign_train_annotated = set()
	std_ign_train_distant = set()
	tot_evidences = 0
	titleset = set([])

	title2vectexSet = {}

	for x in truth:
		title = x['title']
		titleset.add(title)

		vertexSet = x['vertexSet']
		title2vectexSet[title] = vertexSet

		for label in x['labels']:
			r = label['r']

			if use_wikidataid:
				rel = r
				r = rel2id[rel]
			else:
				rel = id2rel[r]

			h_idx = label['h']
			t_idx = label['t']
			std[(title, rel, h_idx, t_idx)] = set(label['evidence'])

			accu_evi.inc_labels(len(label['evidence']))
			accu_evi_by_rel[r].inc_labels(len(label['evidence']))


			accu_re_by_rel[r].inc_labels()

			in_train_annotated = False
			in_train_distant = False
			for n1 in vertexSet[h_idx]:
				for n2 in vertexSet[t_idx]:
					if (n1['name'], n2['name'], rel) in fact_in_train_annotated:
						in_train_annotated = True
					if (n1['name'], n2['name'], rel) in fact_in_train_distant:
						in_train_distant = True

			if not in_train_annotated:
				std_ign_train_annotated.add((title, r, h_idx, t_idx))
				accu_re_ign_train_annotated.inc_labels()
				accu_re_ign_train_annotated_by_rel[r].inc_labels()

			if not in_train_distant:
				std_ign_train_distant.add((title, r, h_idx, t_idx))
				accu_re_ign_train_distant.inc_labels()
				accu_re_ign_train_distant_by_rel[r].inc_labels()

	'''
	for title, r, h_idx, t_idx in std_ign_train_annotated:
		accu_re_ign_train_annotated.inc_labels()
		accu_re_ign_train_annotated_by_rel[r].inc_labels()

	for title, r, h_idx, t_idx in std_ign_train_distant:
		accu_re_ign_train_distant.inc_labels()
		accu_re_ign_train_distant_by_rel[r].inc_labels()
	#'''
	accu_re.inc_labels(len(std))

	correct_in_train_annotated = 0
	correct_in_train_distant = 0
	titleset2 = set([])
	for x in submission_answer:
		if isinstance(x, tuple):
			title, h_idx, t_idx, r = x
		else:
			title = x['title']
			h_idx = x['h_idx']
			t_idx = x['t_idx']
			r = x['r']

		if use_wikidataid:
			rel = r
			r = rel2id[rel]
		else:
			rel = id2rel[r]
		titleset2.add(title)
		if title not in title2vectexSet:
			continue
		vertexSet = title2vectexSet[title]

		if 'evidence' in x:
			evi = set(x['evidence'])
		else:
			evi = set([])

		accu_re.inc_pos()
		accu_re_by_rel[r].inc_pos()
		accu_evi.inc_pos(len(evi))
		accu_evi_by_rel[r].inc_pos(len(evi))


		#'''
		in_train_annotated = in_train_distant = False
		for n1 in vertexSet[h_idx]:
			for n2 in vertexSet[t_idx]:
				if (n1['name'], n2['name'], rel) in fact_in_train_annotated:
					in_train_annotated = True
				if (n1['name'], n2['name'], rel) in fact_in_train_distant:
					in_train_distant = True
		#'''

		if not in_train_annotated:
			accu_re_ign_train_annotated.inc_pos()
			accu_re_ign_train_annotated_by_rel[r].inc_pos()
		if not in_train_distant:
			accu_re_ign_train_distant.inc_pos()
			accu_re_ign_train_distant_by_rel[r].inc_pos()


		if (title, rel, h_idx, t_idx) in std:
			accu_re.inc_tp()
			accu_re_by_rel[r].inc_tp()

			stdevi = std[(title, rel, h_idx, t_idx)]
			correct_evidence = len(stdevi & evi)
			accu_evi.inc_tp(correct_evidence)
			accu_evi_by_rel[r].inc_tp(correct_evidence)


			if in_train_annotated:
				correct_in_train_annotated += 1
			if in_train_distant:
				correct_in_train_distant += 1


			if not in_train_annotated:
				accu_re_ign_train_annotated.inc_tp()
				accu_re_ign_train_annotated_by_rel[r].inc_tp()
			if not in_train_distant:
				accu_re_ign_train_distant.inc_tp()
				accu_re_ign_train_distant_by_rel[r].inc_tp()

	re_p, re_r, re_f1 = accu_re.get_result()
	evi_p, evi_r, evi_f1 = accu_evi.get_result()

	return accu_re, accu_re_ign_train_annotated, accu_re_ign_train_distant, accu_evi,\
		   accu_re_by_rel, accu_re_ign_train_annotated_by_rel, accu_re_ign_train_distant_by_rel, accu_evi_by_rel


def evaluate_by_rel_verbose(test_file, submission_answer, truth, fact_in_train_annotated, fact_in_train_distant, use_wikidataid = True):
	print("Warning: different definition of Ign")
	title2id = {f['title']:i for i, f in enumerate(test_file)}
	rel2id = json.load(open(os.path.join("../src/prepro_data", 'rel2id.json')))
	id2rel = {v: k for k, v in rel2id.items()}

	accu_re = Accuracy()
	accu_re_ign_train_annotated = Accuracy()
	accu_re_ign_train_distant = Accuracy()
	accu_evi = Accuracy()

	accu_re_ign_train_annotated_by_rel = [Accuracy() for _ in range(len(rel2id))]
	accu_re_ign_train_distant_by_rel = [Accuracy() for _ in range(len(rel2id))]
	accu_re_by_rel = [Accuracy() for _ in range(len(rel2id))]
	accu_evi_by_rel = [Accuracy() for _ in range(len(rel2id))]


	std = {}
	std_ign_train_annotated = set()
	std_ign_train_distant = set()
	tot_evidences = 0
	titleset = set([])

	title2vectexSet = {}

	for x in truth:
		title = x['title']
		titleset.add(title)

		vertexSet = x['vertexSet']
		title2vectexSet[title] = vertexSet

		for label in x['labels']:
			r = label['r']

			if use_wikidataid:
				rel = r
				r = rel2id[rel]
			else:
				rel = id2rel[r]

			h_idx = label['h']
			t_idx = label['t']
			std[(title, rel, h_idx, t_idx)] = set(label['evidence'])

			accu_evi.inc_labels(len(label['evidence']))
			accu_evi_by_rel[r].inc_labels(len(label['evidence']))


			accu_re_by_rel[r].inc_labels()

			in_train_annotated = False
			in_train_distant = False
			for n1 in vertexSet[h_idx]:
				for n2 in vertexSet[t_idx]:
					if (n1['name'], n2['name'], rel) in fact_in_train_annotated:
						in_train_annotated = True
					if (n1['name'], n2['name'], rel) in fact_in_train_distant:
						in_train_distant = True

			if not in_train_annotated:
				std_ign_train_annotated.add((title, r, h_idx, t_idx))
				accu_re_ign_train_annotated.inc_labels()
				accu_re_ign_train_annotated_by_rel[r].inc_labels()

			if not in_train_distant:
				std_ign_train_distant.add((title, r, h_idx, t_idx))
				accu_re_ign_train_distant.inc_labels()
				accu_re_ign_train_distant_by_rel[r].inc_labels()

	'''
	for title, r, h_idx, t_idx in std_ign_train_annotated:
		accu_re_ign_train_annotated.inc_labels()
		accu_re_ign_train_annotated_by_rel[r].inc_labels()

	for title, r, h_idx, t_idx in std_ign_train_distant:
		accu_re_ign_train_distant.inc_labels()
		accu_re_ign_train_distant_by_rel[r].inc_labels()
	#'''
	accu_re.inc_labels(len(std))

	correct_in_train_annotated = 0
	correct_in_train_distant = 0
	titleset2 = set([])
	for x in submission_answer:
		if isinstance(x, tuple):
			title, h_idx, t_idx, r = x
		else:
			title = x['title']
			h_idx = x['h_idx']
			t_idx = x['t_idx']
			r = x['r']

		if use_wikidataid:
			rel = r
			r = rel2id[rel]
		else:
			rel = id2rel[r]

		if rel == "P190" or rel == "P36":
			doc = test_file[title2id[title]]
			vertexSet = doc['vertexSet']
			max_sent = 0

			max_sent = max(max_sent, max(vertexSet[h_idx][0]['sent_id'], vertexSet[t_idx][0]['sent_id']))
			if max_sent >1:
				continue
			print(f"{title2id[title]} Doc:({title})===================================")

			print(vertexSet[h_idx][0]['name'], rel, vertexSet[t_idx][0]['name'])

			for sid in range(max_sent + 1):
				print(sid, " ".join(doc['sents'][sid]))

		titleset2.add(title)
		if title not in title2vectexSet:
			continue
		vertexSet = title2vectexSet[title]

		if 'evidence' in x:
			evi = set(x['evidence'])
		else:
			evi = set([])

		accu_re.inc_pos()
		accu_re_by_rel[r].inc_pos()
		accu_evi.inc_pos(len(evi))
		accu_evi_by_rel[r].inc_pos(len(evi))


		#'''
		in_train_annotated = in_train_distant = False
		for n1 in vertexSet[h_idx]:
			for n2 in vertexSet[t_idx]:
				if (n1['name'], n2['name'], rel) in fact_in_train_annotated:
					in_train_annotated = True
				if (n1['name'], n2['name'], rel) in fact_in_train_distant:
					in_train_distant = True
		#'''

		if not in_train_annotated:
			accu_re_ign_train_annotated.inc_pos()
			accu_re_ign_train_annotated_by_rel[r].inc_pos()
		if not in_train_distant:
			accu_re_ign_train_distant.inc_pos()
			accu_re_ign_train_distant_by_rel[r].inc_pos()


		if (title, rel, h_idx, t_idx) in std:
			accu_re.inc_tp()
			accu_re_by_rel[r].inc_tp()

			stdevi = std[(title, rel, h_idx, t_idx)]
			correct_evidence = len(stdevi & evi)
			accu_evi.inc_tp(correct_evidence)
			accu_evi_by_rel[r].inc_tp(correct_evidence)


			if in_train_annotated:
				correct_in_train_annotated += 1
			if in_train_distant:
				correct_in_train_distant += 1


			if not in_train_annotated:
				accu_re_ign_train_annotated.inc_tp()
				accu_re_ign_train_annotated_by_rel[r].inc_tp()
			if not in_train_distant:
				accu_re_ign_train_distant.inc_tp()
				accu_re_ign_train_distant_by_rel[r].inc_tp()

	re_p, re_r, re_f1 = accu_re.get_result()
	evi_p, evi_r, evi_f1 = accu_evi.get_result()

	return accu_re, accu_re_ign_train_annotated, accu_re_ign_train_distant, accu_evi,\
		   accu_re_by_rel, accu_re_ign_train_annotated_by_rel, accu_re_ign_train_distant_by_rel, accu_evi_by_rel

def evaluate(submission_answer, truth, fact_in_train_annotated, fact_in_train_distant, use_wikidataid = True):
	if not use_wikidataid:
		rel2id = json.load(open(os.path.join("../src/prepro_data", 'rel2id.json')))
		id2rel = {v: k for k, v in rel2id.items()}
	std = {}
	tot_evidences = 0
	titleset = set([])


	title2vectexSet = {}

	for x in truth:
		title = x['title']
		titleset.add(title)

		vertexSet = x['vertexSet']
		title2vectexSet[title] = vertexSet

		for label in x['labels']:
			r = label['r']
			if not use_wikidataid:
				r = id2rel[r]

			h_idx = label['h']
			t_idx = label['t']
			std[(title, r, h_idx, t_idx)] = set(label['evidence'])
			tot_evidences += len(label['evidence'])

	tot_relations = len(std)


	correct_re = 0
	correct_evidence = 0
	pred_evi = 0

	correct_in_train_annotated = 0
	correct_in_train_distant = 0
	titleset2 = set([])
	for x in submission_answer:
		if isinstance(x, tuple):
			title, h_idx, t_idx, r = x
		else:
			title = x['title']
			h_idx = x['h_idx']
			t_idx = x['t_idx']
			r = x['r']

		if not use_wikidataid:
			r = id2rel[r]
		titleset2.add(title)
		if title not in title2vectexSet:
			continue
		vertexSet = title2vectexSet[title]

		if 'evidence' in x:
			evi = set(x['evidence'])
		else:
			evi = set([])
		pred_evi += len(evi)

		if (title, r, h_idx, t_idx) in std:
			correct_re += 1
			stdevi = std[(title, r, h_idx, t_idx)]
			correct_evidence += len(stdevi & evi)
			in_train_annotated = in_train_distant = False
			for n1 in vertexSet[h_idx]:
				for n2 in vertexSet[t_idx]:
					if (n1['name'], n2['name'], r) in fact_in_train_annotated:
						in_train_annotated = True
					if (n1['name'], n2['name'], r) in fact_in_train_distant:
						in_train_distant = True

			if in_train_annotated:
				correct_in_train_annotated += 1
			if in_train_distant:
				correct_in_train_distant += 1

	if len(submission_answer) == 0:
		etc_result = ((0.0, 0.0), 0, 0)
		return 0.0, 0.0, 0.0, 0.0, etc_result


	re_p = 1.0 * correct_re / len(submission_answer)
	re_r = 1.0 * correct_re / tot_relations
	if re_p + re_r == 0:
		re_f1 = 0
	else:
		re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

	evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
	evi_r = 1.0 * correct_evidence / tot_evidences
	if evi_p + evi_r == 0:
		evi_f1 = 0
	else:
		evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

	re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (
			len(submission_answer) - correct_in_train_annotated)
	re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (
			len(submission_answer) - correct_in_train_distant)

	if re_p_ignore_train_annotated + re_r == 0:
		re_f1_ignore_train_annotated = 0
	else:
		re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

	if re_p_ignore_train + re_r == 0:
		re_f1_ignore_train = 0
	else:
		re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

	etc_result = ((re_p, re_r),correct_re, correct_evidence)

	return re_f1, evi_f1, re_f1_ignore_train, re_f1_ignore_train_annotated, etc_result

if __name__ == "__main__":
	if len(sys.argv) == 1:
		argv = ["../eval","../eval"]
	else:
		argv = sys.argv[1:]

	input_dir = argv[0]
	output_dir = argv[1]

	submit_dir = os.path.join(input_dir, 'res')
	truth_dir = os.path.join(input_dir, 'ref')

	if not os.path.isdir(submit_dir):
		print("%s doesn't exist" % submit_dir)

	if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		fact_in_train_annotated = gen_train_facts("../data/train_annotated.json", truth_dir)
		fact_in_train_distant = gen_train_facts("../data/train_distant.json", truth_dir)

		output_filename = os.path.join(output_dir, 'scores.txt')
		output_file = open(output_filename, 'w')

		truth_file = os.path.join(truth_dir, truth_file_name)
		truth = json.load(open(truth_file))

		submission_answer_file = os.path.join(submit_dir, "result.json")
		submission_answer = json.load(open(submission_answer_file))

		re_f1, evi_f1, re_f1_ignore_train, re_f1_ignore_train_annotated, etc_result \
			= evaluate(submission_answer, truth, fact_in_train_annotated, fact_in_train_distant, use_wikidataid = True)

		print('RE_F1:', re_f1)
		print('Evi_F1:', evi_f1)
		print('RE_ignore_annotated_F1:', re_f1_ignore_train_annotated)
		print('RE_ignore_distant_F1:', re_f1_ignore_train)

		output_file.write("RE_F1: %f\n" % re_f1)
		output_file.write("Evi_F1: %f\n" % evi_f1)

		output_file.write("RE_ignore_annotated_F1: %f\n" % re_f1_ignore_train_annotated)
		output_file.write("RE_ignore_distant_F1: %f\n" % re_f1_ignore_train)

		output_file.close()

