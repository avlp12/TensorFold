"""Changed integration/ABI only: CPU tensors and original-kernel substitutes."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
import unittest
import torch

torch.set_default_device('cpu')
torch.set_num_threads(2)
p = Path(__file__).parents[1] / 'src/tensorfold/families/glm5_next/cuda/b0_speculation.py'
spec = importlib.util.spec_from_file_location('b0spec', p)
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)


def weight(vocab=8):
    return SimpleNamespace(world=1, rank=0, vocab_offset=0, cfg=SimpleNamespace(vocab=vocab, eos=[7]), meta={}, comm=None)


class Integration(unittest.TestCase):
    def modules(self, fn, events):
        mods = {}
        def module(name, **functions):
            obj = ModuleType(name); obj.__dict__.update(functions); mods[name] = obj
        module('vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils', rejection_sample=fn)
        def temp(logits, mapping, temperatures):
            events.append('temperature'); logits.div_(temperatures[0])
        def minp(logits, mapping, values): events.append('min_p')
        def top(logits, k, p): events.append('top_k_top_p'); return logits
        module('vllm.v1.worker.gpu.sample.gumbel', apply_temperature=temp)
        module('vllm.v1.worker.gpu.sample.min_p', apply_min_p=minp)
        module('vllm.v1.sample.ops.topk_topp_sampler', apply_top_k_top_p=top)
        return mods

    def test_original_rejection_result_and_shortened_conditional_q(self):
        events=[]
        proposal=b.Proposal([3,4,5], torch.tensor([[2,3],[4,6],[1,5]]), torch.tensor([[0.,2.],[1.,3.],[4.,5.]]))
        logits=torch.arange(24,dtype=torch.float32).view(3,8)
        original=logits.clone()
        def reject(*a, **kw):
            self.assertEqual(events,['temperature','min_p','top_k_top_p'])
            self.assertTrue(torch.equal(a[0],original/2))
            self.assertEqual(a[1].shape,(1,3,8))
            self.assertEqual(a[1][0,0,3].item(),2.)
            self.assertEqual(a[1][0,1,6].item(),3.)
            self.assertTrue(torch.isneginf(a[1][0,2]).all())
            self.assertEqual(a[2].tolist(),[1,3,4])
            self.assertEqual(a[3].tolist(),[0,3])
            self.assertEqual(a[4].tolist(),[20,21,22])
            self.assertEqual(a[7].tolist(),[0,1,2])
            self.assertEqual(a[10],3)
            self.assertEqual(kw,dict(synthetic_conditional_rates=None,use_fp64=False,use_block_verification=False))
            # Original kernel accepted 3 then resampled 7. No keyed-match loop.
            return torch.tensor([[3,7,-1,-1]]),torch.tensor([2])
        smp=SimpleNamespace(seed=42,temperature=2.,min_p=.1,top_k=4,top_p=.9)
        with patch.dict('sys.modules',self.modules(reject,events)):
            result=b.verify(weight(),logits,[1,3,4],proposal,20,smp)
        self.assertEqual(result,[3,7]);self.assertTrue(torch.equal(logits,original))

    def test_short_round_and_eos_stop(self):
        def reject(*a,**kw):
            self.assertTrue(torch.isneginf(a[1]).all())
            self.assertEqual(a[3].tolist(),[0,1])
            return torch.tensor([[7,-1,-1,-1]]),torch.tensor([1])
        with patch.dict('sys.modules',self.modules(reject,[])):
            self.assertEqual(b.verify(weight(),torch.zeros(1,8),[1],None,0,None,stop_eos=True),[7])

    def test_missing_or_mismatched_proposal_is_rejected(self):
        with patch.dict('sys.modules',self.modules(lambda *a: self.fail('kernel must not run'),[])):
            for q in [None, b.Proposal([5],torch.tensor([[5]]),torch.ones(1,1))]:
                with self.assertRaisesRegex(ValueError,'do not match'):
                    b.verify(weight(),torch.zeros(2,8),[1,3],q,0,None)

    def test_unequal_tp_shards(self):
        left=torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        right=torch.tensor([[7.,8.],[9.,10.]])
        def gather(src,dst):
            if src.dtype==torch.int64:
                dst.copy_(torch.tensor([0,3,3,2]));return
            pad=torch.full((2,3),-float('inf'));pad[:,:2]=right
            dst.copy_(torch.cat([left.flatten(),pad.flatten()]))
        w=weight(5);w.world=2;w.comm=SimpleNamespace(all_gather=gather)
        self.assertTrue(torch.equal(b.full_logits(w,left),torch.cat([left,right],dim=1)))
        self.assertEqual(w.meta['b0_rejection_vocab_layout'],[[0,3],[3,2]])


if __name__=='__main__':unittest.main()
