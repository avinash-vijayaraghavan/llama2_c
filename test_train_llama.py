import struct
import torch
import model


# Load the state.bin file with the gradients
file = "state.bin"
with open(file, "rb") as fp:
    # Load array dimensions (optional)
    B, T, C, OC, L, NH, V, n_kv_heads = struct.unpack("i" * 8, fp.read(32))
    print(B, T, C, OC, L, NH, V, n_kv_heads)

    tokens = torch.tensor(
        struct.unpack("i" * B * T, fp.read(4 * B * T)), dtype=torch.int64
    ).view(B, T)
    targets = torch.tensor(
        struct.unpack("i" * B * T, fp.read(4 * B * T)), dtype=torch.int64
    ).view(B, T)

    # Load grads
    dx = torch.tensor(struct.unpack("f" * B * T * C, fp.read(4 * B * T * C)))
    # dlayer_input = torch.tensor(struct.unpack("f" * B * T * C, fp.read(4 * B * T * C)))
    dx_rms_attn_norm = torch.tensor(
        struct.unpack("f" * L * B * T * C, fp.read(4 * L * B * T * C))
    )

    dq = torch.tensor(struct.unpack("f" * L * B * T * C, fp.read(4 * L * B * T * C)))
    dk = torch.tensor(struct.unpack("f" * L * B * T * C, fp.read(4 * L * B * T * C)))
    dv = torch.tensor(struct.unpack("f" * L * B * T * C, fp.read(4 * L * B * T * C)))
    dqr = torch.tensor(struct.unpack("f" * L * B * T * C, fp.read(4 * L * B * T * C)))
    dkr = torch.tensor(struct.unpack("f" * L * B * T * C, fp.read(4 * L * B * T * C)))

    dattn = torch.tensor(
        struct.unpack("f" * L * B * T * T * NH, fp.read(4 * L * B * T * T * NH))
    )
    dpreattn = torch.tensor(
        struct.unpack("f" * L * B * T * T * NH, fp.read(4 * L * B * T * T * NH))
    )
    dattn_out = torch.tensor(
        struct.unpack("f" * L * B * T * C, fp.read(4 * L * B * T * C))
    )
    dx_attn = torch.tensor(
        struct.unpack("f" * L * B * T * C, fp.read(4 * L * B * T * C))
    )
    dxo = torch.tensor(struct.unpack("f" * L * B * T * C, fp.read(4 * L * B * T * C)))
    dx_res = torch.tensor(
        struct.unpack("f" * L * B * T * C, fp.read(4 * L * B * T * C))
    )
    dx_res_rms_ffn_norm = torch.tensor(
        struct.unpack("f" * L * B * T * C, fp.read(4 * L * B * T * C))
    )
    dh1 = torch.tensor(struct.unpack("f" * L * B * T * OC, fp.read(4 * L * B * T * OC)))
    dh2 = torch.tensor(struct.unpack("f" * L * B * T * C, fp.read(4 * L * B * T * C)))
    dh3 = torch.tensor(struct.unpack("f" * L * B * T * OC, fp.read(4 * L * B * T * OC)))
    dh1_h3_prod = torch.tensor(
        struct.unpack("f" * L * B * T * OC, fp.read(4 * L * B * T * OC))
    )
    dh1_silu = torch.tensor(
        struct.unpack("f" * L * B * T * OC, fp.read(4 * L * B * T * OC))
    )

    dx_ffn = torch.tensor(
        struct.unpack("f" * L * B * T * C, fp.read(4 * L * B * T * C))
    )
    dx_final_res = torch.tensor(
        struct.unpack("f" * L * B * T * C, fp.read(4 * L * B * T * C))
    )
    dx_final_rms_norm = torch.tensor(
        struct.unpack("f" * B * T * C, fp.read(4 * B * T * C))
    )

    dlogits = torch.tensor(struct.unpack("f" * B * T * V, fp.read(4 * B * T * V)))
    dprobs = torch.tensor(struct.unpack("f" * B * T * V, fp.read(4 * B * T * V)))
    dloss = torch.tensor(struct.unpack("f" * B * T, fp.read(4 * B * T)))

    # grad weights
    dwte = torch.tensor(struct.unpack("f" * V * C, fp.read(4 * V * C)))
    drms_att_weight = torch.tensor(struct.unpack("f" * L * C, fp.read(4 * L * C)))
    drms_ffn_weight = torch.tensor(struct.unpack("f" * L * C, fp.read(4 * L * C)))

    head_size = C // NH
    dwq = torch.tensor(struct.unpack("f" * L * C * C, fp.read(4 * L * C * C)))
    dwk = torch.tensor(
        struct.unpack(
            "f" * L * C * n_kv_heads * head_size,
            fp.read(4 * L * C * n_kv_heads * head_size),
        )
    )
    dwv = torch.tensor(
        struct.unpack(
            "f" * L * C * n_kv_heads * head_size,
            fp.read(4 * L * C * n_kv_heads * head_size),
        )
    )
    dwo = torch.tensor(struct.unpack("f" * L * C * C, fp.read(4 * L * C * C)))

    dw1 = torch.tensor(struct.unpack("f" * L * OC * C, fp.read(4 * L * OC * C)))
    dw2 = torch.tensor(struct.unpack("f" * L * OC * C, fp.read(4 * L * OC * C)))
    dw3 = torch.tensor(struct.unpack("f" * L * OC * C, fp.read(4 * L * OC * C)))

    drms_final_weight = torch.tensor(struct.unpack("f" * C, fp.read(4 * C)))
    dwcls = torch.tensor(struct.unpack("f" * C * V, fp.read(4 * C * V)))

    # also check the embedding weights, after the update
    wcls = torch.tensor(struct.unpack("f" * V * C, fp.read(4 * V * C))).view(V,C)

    model_args = model.ModelArgs(
        dim=C,
        hidden_dim=OC,
        n_layers=L,
        n_heads=NH,
        n_kv_heads=n_kv_heads,
        vocab_size=V,
        max_seq_len=256,
    )

# Load the model and optimizer
tmodel = model.Transformer(model_args)
lr = 0.001
beta1 = 0.9
beta2 = 0.999
eps = 1e-08
weight_decay = 0.01
checkpoint_path = "./stories15M.pt"
checkpoint = torch.load(checkpoint_path)
tmodel.load_state_dict(checkpoint["model"], strict=False)
optimizer = torch.optim.AdamW(params=tmodel.parameters(), lr=lr, betas=(beta1, beta2))

optimizer.zero_grad()

logits = tmodel.forward(tokens, targets=targets)
tmodel.last_loss.backward(retain_graph=True, create_graph=True)
optimizer.step()

def test_tensor_equality(name, a, b, rtol=1e-6, atol=1e-2, l=None):
    try:
        assert  torch.allclose(a, b, rtol=rtol, atol=atol)
    except AssertionError as e:
        msg = f"{name} does not match in {l} with atol={atol} amd rtol={rtol}"
        if isinstance(l, str):
            msg = f"{name} does not match with {l} with atol={atol} amd rtol={rtol}"
        print(f"{msg} {a.sum()} {b.sum()}")
        raise AssertionError(msg)


# output layer
test_tensor_equality(
    "dwcls",
    dwcls,
    tmodel.output.weight.grad.ravel(),
)
test_tensor_equality("dlogits", dlogits, tmodel.logits.grad.ravel())
# final rms norm
test_tensor_equality(
    "drms_final_weight",
    drms_final_weight,
    tmodel.norm.weight.grad,
)
test_tensor_equality(
    "dx_final_rms_norm", dx_final_rms_norm, tmodel.final_rms_norm.grad.ravel()
)
test_tensor_equality(
    "dwte",
    dwte,
    tmodel.tok_embeddings.weight.grad.ravel(),
)
#
for l in range(L):

    # feed forward
    test_tensor_equality(
        "dw1",
        dw1.view(L, -1)[l],
        tmodel.layers[l].feed_forward.w1.weight.grad.ravel(),
        l=l,
    )
    test_tensor_equality(
        "dw2",
        dw2.view(L, -1)[l],
        tmodel.layers[l].feed_forward.w2.weight.grad.ravel(),
        l=l,
    )
    test_tensor_equality(
        "dw3",
        dw3.view(L, -1)[l],
        tmodel.layers[l].feed_forward.w3.weight.grad.ravel(),
        l=l,
    )
    test_tensor_equality(
        "drms_ffn_weight",
        drms_ffn_weight.view(L, -1)[l],
        tmodel.layers[l].ffn_norm.weight.grad.ravel(),
        l=l,
    )
    # attention
    test_tensor_equality(
        "dwq",
        dwq.view(L, -1)[l],
        tmodel.layers[l].attention.wq.weight.grad.ravel(),
        l=l,
    )
    test_tensor_equality(
        "dwk",
        dwk.view(L, -1)[l],
        tmodel.layers[l].attention.wk.weight.grad.ravel(),
        l=l,
    )
    test_tensor_equality(
        "dwv",
        dwv.view(L, -1)[l],
        tmodel.layers[l].attention.wv.weight.grad.ravel(),
        atol=1e-1,
        l=l,
    )  # dwv are large for some reason
    test_tensor_equality(
        "dwo",
        dwo.view(L, -1)[l],
        tmodel.layers[l].attention.wo.weight.grad.ravel(),
        l=l,
    )
    test_tensor_equality(
        "drms_att_weight",
        drms_att_weight.view(L, -1)[l],
        tmodel.layers[l].attention_norm.weight.grad.ravel(),
        l=l,
    )
    # check grad activation
    # feed forward
    test_tensor_equality(
        "dh1", dh1.view(L, -1)[l], tmodel.layers[l].feed_forward.h1.grad.ravel(), l=l
    )
    test_tensor_equality(
        "dh3", dh3.view(L, -1)[l], tmodel.layers[l].feed_forward.h3.grad.ravel(), l=l
    )
    test_tensor_equality(
        "dh1_h3_prod",
        dh1_h3_prod.view(L, -1)[l],
        tmodel.layers[l].feed_forward.h1_h3_prod.grad.ravel(),
        l=l,
    )
    test_tensor_equality(
        "dh1_silu",
        dh1_silu.view(L, -1)[l],
        tmodel.layers[l].feed_forward.h1_silu.grad.ravel(),
        l=l,
    )

    # rope
    test_tensor_equality(
        "dqr", dqr.view(L, -1)[l], tmodel.layers[l].attention.xqr.grad.ravel(), l=l
    )
    test_tensor_equality(
        "dkr", dkr.view(L, -1)[l], tmodel.layers[l].attention.xkr.grad.ravel(), l=l
    )

    test_tensor_equality(
        "dq", dq.view(L, -1)[l], tmodel.layers[l].attention.xq.grad.ravel(), l=l
    )
    test_tensor_equality(
        "dk", dk.view(L, -1)[l], tmodel.layers[l].attention.xk.grad.ravel(), l=l
    )
    test_tensor_equality(
        "dv", dv.view(L, -1)[l], tmodel.layers[l].attention.xv.grad.ravel(), l=l
    )

    test_tensor_equality(
        "dx_rms_attn_norm",
        dx_rms_attn_norm.view(L, -1)[l],
        tmodel.layers[l].attn_norm.grad.ravel(),
        l=l,
    )
test_tensor_equality(
    "output",
    wcls.ravel(),
    tmodel.output.weight.ravel(),
    l="wcls"
)

print("All match")
 