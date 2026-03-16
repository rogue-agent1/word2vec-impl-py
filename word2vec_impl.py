#!/usr/bin/env python3
"""Word2Vec skip-gram with negative sampling."""
import random, math
from collections import Counter

class Word2Vec:
    def __init__(self, dim=10, window=2, neg=5, lr=0.025, epochs=5):
        self.dim=dim;self.window=window;self.neg=neg;self.lr=lr;self.epochs=epochs
    def fit(self, sentences):
        words=[];[words.extend(s) for s in sentences]
        freq=Counter(words);self.vocab={w:i for i,w in enumerate(freq)}
        V=len(self.vocab);self.W=[[random.gauss(0,0.1) for _ in range(self.dim)] for _ in range(V)]
        self.C=[[0]*self.dim for _ in range(V)]
        # Negative sampling table
        table=[];[table.extend([self.vocab[w]]*int(freq[w]**0.75)) for w in freq]
        for _ in range(self.epochs):
            for sent in sentences:
                ids=[self.vocab[w] for w in sent]
                for i,w in enumerate(ids):
                    for j in range(max(0,i-self.window),min(len(ids),i+self.window+1)):
                        if i==j: continue
                        c=ids[j]
                        # Positive
                        dot=sum(self.W[w][d]*self.C[c][d] for d in range(self.dim))
                        sig=1/(1+math.exp(-max(-10,min(10,dot))))
                        grad=self.lr*(1-sig)
                        for d in range(self.dim):
                            self.W[w][d]+=grad*self.C[c][d];self.C[c][d]+=grad*self.W[w][d]
                        # Negative
                        for _ in range(self.neg):
                            neg=table[random.randint(0,len(table)-1)]
                            dot=sum(self.W[w][d]*self.C[neg][d] for d in range(self.dim))
                            sig=1/(1+math.exp(-max(-10,min(10,dot))))
                            grad=self.lr*(-sig)
                            for d in range(self.dim):
                                self.W[w][d]+=grad*self.C[neg][d];self.C[neg][d]+=grad*self.W[w][d]
    def vector(self,word): return self.W[self.vocab[word]] if word in self.vocab else None
    def similarity(self,w1,w2):
        v1,v2=self.vector(w1),self.vector(w2)
        if not v1 or not v2: return 0
        dot=sum(a*b for a,b in zip(v1,v2))
        n1=math.sqrt(sum(a*a for a in v1));n2=math.sqrt(sum(b*b for b in v2))
        return dot/(n1*n2) if n1*n2>0 else 0

def main():
    sents=[["the","cat","sat"],["the","dog","ran"],["a","cat","ran"],["the","bird","flew"]]
    w2v=Word2Vec(dim=5,epochs=10);w2v.fit(sents)
    print(f"cat-dog similarity: {w2v.similarity('cat','dog'):.4f}")

if __name__=="__main__":main()
