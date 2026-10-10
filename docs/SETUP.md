# Cloud setup and cleanup

These commands are extracted from codelab steps 2-4 and 19. Run them deliberately
in the named environment; provisioning is not performed by installing the Python
package. Refer to the [source tutorial](https://codelabs.developers.google.com/alloydb-agentic-tiered-memory)
for prerequisites, organization policies, and regional resource availability.

For an existing deployment, obtain its project, endpoints, and database configuration
from the infrastructure owner. Each developer still needs their own IAM database
user (`DB_USER`), Application Default Credentials, and exported environment variables.
The private endpoints require VPC connectivity; the codelab uses a VM in that VPC.

Replace `<YOUR_PROJECT_ID>` and other placeholders before executing commands.
All provisioning and teardown commands below are shown for reference and are not
run by CI. Region and resource names are examples from the codelab.

## Cloud Shell: choose a project and enable APIs

```bash
gcloud auth list
```

```bash
export PROJECT_ID=<YOUR_PROJECT_ID>
gcloud config set project $PROJECT_ID

export REGION=us-east1
export GENAI_LOCATION=us
export ZONE=us-east1-b
export ADBCLUSTER=agent-memory-cluster
export ADBINSTANCE=agent-memory-instance
export VALKEYINSTANCE=agent-memory-cache
export VM_NAME=agent-dev-vm
```

```bash
gcloud services enable \
  alloydb.googleapis.com \
  memorystore.googleapis.com \
  aiplatform.googleapis.com \
  compute.googleapis.com \
  servicenetworking.googleapis.com \
  networkconnectivity.googleapis.com
```

```bash
gcloud compute instances create $VM_NAME \
    --zone=$ZONE \
    --machine-type=e2-standard-2 \
    --scopes=cloud-platform \
    --network=default \
    --shielded-secure-boot
```

## Cloud Shell: provision private access, AlloyDB, and Valkey

```bash
gcloud compute addresses create psa-range \
    --global \
    --purpose=VPC_PEERING \
    --prefix-length=24 \
    --description="VPC private service access" \
    --network=default
```

```bash
gcloud services vpc-peerings connect \
    --service=servicenetworking.googleapis.com \
    --ranges=psa-range \
    --network=default
```

```bash
export PGPASSWORD=`openssl rand -hex 12`
```

```bash
gcloud alloydb clusters create $ADBCLUSTER \
    --password=$PGPASSWORD \
    --network=default \
    --region=$REGION \
    --subscription-type=TRIAL
```

```bash
gcloud alloydb instances create $ADBINSTANCE \
    --instance-type=PRIMARY \
    --cpu-count=2 \
    --region=$REGION \
    --cluster=$ADBCLUSTER
```

```bash
gcloud network-connectivity service-connection-policies create memorystore-policy \
    --network=default \
    --region=$REGION \
    --service-class=gcp-memorystore \
    --subnets=projects/$PROJECT_ID/regions/$REGION/subnetworks/default
```

```bash
gcloud memorystore instances create $VALKEYINSTANCE \
    --location=$REGION \
    --shard-count=1 \
    --replica-count=0 \
    --node-type=SHARED_CORE_NANO \
    --psc-auto-connections="network=projects/$PROJECT_ID/global/networks/default,projectId=$PROJECT_ID"
```

```bash
PROJECT_ID=$(gcloud config get-value project)

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:service-$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")@gcp-sa-alloydb.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

If your AlloyDB trial has already been used, the codelab specifies `--subscription-type=STANDARD` instead.

## Cloud Shell: configure IAM database access and retrieve endpoints

```bash
gcloud beta alloydb instances update $ADBINSTANCE \
  --cluster=$ADBCLUSTER \
  --region=$REGION \
  --update-mode=FORCE_APPLY \
  --database-flags=alloydb.iam_authentication=on,google_ml_integration.enable_ai_query_engine=on
```

```bash
export USER_ACCOUNT=$(gcloud config get-value account)

gcloud alloydb users create $USER_ACCOUNT \
    --cluster=$ADBCLUSTER \
    --region=$REGION \
    --type=IAM_BASED \
    --db-roles=alloydbsuperuser
```

```bash
# Export GCP Project ID, Region, and IAM User
export PROJECT_ID=$(gcloud config get-value project)
export REGION=us-east1
export DB_USER=$(gcloud config get-value account)

# Retrieve Internal VPC IP Addresses for AlloyDB & Memorystore for Valkey
export DB_HOST=$(gcloud alloydb instances describe $ADBINSTANCE --cluster=$ADBCLUSTER --region=$REGION --format="value(ipAddress)")
export DB_PORT=5432
export DB_NAME=postgres

export VALKEY_HOST=$(gcloud memorystore instances describe $VALKEYINSTANCE --location=$REGION --format="value(discoveryEndpoints[0].address)")
export VALKEY_PORT=6379

# Verify exported endpoints
echo "Project ID:        $PROJECT_ID"
echo "Region:            $REGION"
echo "AlloyDB Private IP: $DB_HOST"
echo "IAM DB User:       $DB_USER"
echo "Valkey Private IP:  $VALKEY_HOST"
```

## Cloud Shell: connect to the development VM

```bash
gcloud compute ssh $VM_NAME --zone=$ZONE
```

```bash
gcloud compute firewall-rules create allow-ssh \
  --network=default --allow=tcp:22 --source-ranges=35.235.240.0/20 \
  --priority=100 --project=${PROJECT_ID}
```

The firewall command is only needed if direct SSH fails. Use IAP with:

```bash
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --tunnel-through-iap
```

## Development VM: export your own connection settings

```bash
export PROJECT_ID=<YOUR_PROJECT_ID>
export REGION=us-east1
export GENAI_LOCATION=us
export DB_HOST=<YOUR_ALLOYDB_PRIVATE_IP>
export DB_PORT=5432
export DB_NAME=postgres
export DB_USER=<YOUR_IAM_DB_USER>
export VALKEY_HOST=<YOUR_VALKEY_PRIVATE_IP>
export VALKEY_PORT=6379
```

## Development VM: authentication

Clone this repository on the VM, install it as described in the root README,
then authenticate:

```bash
gcloud auth application-default login
```

Initialize the [SQL schema](../sql/README.md) in AlloyDB Studio, then run the demos
from the repository root. The codelab models must be available to your project.
The LLM model identifier is used by the Python examples and SQL registration;
if adapting it, change all call sites and the registered endpoint together.

## Cleanup

`cleanup_memory_system.py` resets demo data: it calls `FLUSHDB` on the configured
Valkey database and truncates all four demo memory/permission tables. Use a
dedicated demo database when running that script.

```bash
uv run python cleanup_memory_system.py
```

For cloud teardown, return to Cloud Shell and verify the selected project and
resource names before deleting:

```bash
# Delete Compute Engine development VM
gcloud compute instances delete $VM_NAME \
  --zone=$ZONE \
  --quiet

# Delete AlloyDB primary instance and cluster
gcloud alloydb instances delete $ADBINSTANCE \
  --cluster=$ADBCLUSTER \
  --region=$REGION \
  --quiet

gcloud alloydb clusters delete $ADBCLUSTER \
  --region=$REGION \
  --quiet

# Delete Memorystore for Valkey instance
gcloud memorystore instances delete $VALKEYINSTANCE \
  --location=$REGION \
  --quiet
```

The source teardown removes the main VM, database, and cache resources. It
does not remove the optional firewall rule, private service access allocation,
peering connection, or service connection policy. Review those separately if
they are dedicated to this demo; they can also be shared by other resources.
