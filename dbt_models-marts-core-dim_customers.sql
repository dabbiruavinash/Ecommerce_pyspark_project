# DBT Mode

{{
  config(
    materialized='incremental',
    incremental_strategy='merge',
    unique_key='customer_id',
    schema='analytics',
    tags=['core', 'customers'],
    grants={'select': ['analyst', 'bi_tool']},
    persist_docs={'columns': true}
  )
}}

with customer_source as (
    select * from {{ source('ecommerce', 'customers') }}
),

customer_orders as (
    select
        customer_id,
        min(order_date) as first_order_date,
        max(order_date) as most_recent_order_date,
        count(order_id) as number_of_orders,
        sum(total_amount) as lifetime_value
    from {{ ref('stg_orders') }}
    group by 1
),

customer_payments as (
    select
        customer_id,
        sum(case when payment_status = 'completed' then amount else 0 end) as total_paid,
        sum(case when payment_status = 'failed' then amount else 0 end) as total_failed,
        count(distinct case when payment_status = 'completed' then payment_id end) as successful_payments_count
    from {{ ref('stg_payments') }}
    group by 1
),

customer_devices as (
    select
        customer_id,
        count(distinct device_id) as devices_used,
        mode(device_type) as most_common_device
    from {{ ref('stg_sessions') }}
    group by 1
)

select
    c.customer_id,
    c.first_name,
    c.last_name,
    c.email,
    c.phone,
    c.registration_date,
    c.loyalty_tier,
    c.demographics,
    c.preferences,
    o.first_order_date,
    o.most_recent_order_date,
    o.number_of_orders,
    o.lifetime_value,
    p.total_paid,
    p.total_failed,
    p.successful_payments_count,
    d.devices_used,
    d.most_common_device,
    case
        when o.lifetime_value > 1000 then 'platinum'
        when o.lifetime_value > 500 then 'gold'
        when o.lifetime_value > 100 then 'silver'
        else 'bronze'
    end as value_segment,
    {{ dbt_utils.datediff('o.first_order_date', 'current_date', 'day') }} as customer_tenure_days,
    {{ dbt_utils.datediff('o.most_recent_order_date', 'current_date', 'day') }} as days_since_last_order
from customer_source c
left join customer_orders o on c.customer_id = o.customer_id
left join customer_payments p on c.customer_id = p.customer_id
left join customer_devices d on c.customer_id = d.customer_id

{% if is_incremental() %}
  where c.updated_at > (select max(updated_at) from {{ this }})
{% endif %}