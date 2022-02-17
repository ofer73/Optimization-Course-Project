import os

bashcmd = os.popen('bash s.sh').readlines()
for l in bashcmd[1:]:
	jid = [i for i in l.split(" ") if i]
	try:
		name, jid = jid[0], int(jid[1])
	except Exception:
		continue

	if name.startswith("optim"):
		print(name)
		os.system(f'scancel {jid}')

